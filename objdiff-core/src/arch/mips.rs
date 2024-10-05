use std::{borrow::Cow, collections::BTreeMap, sync::Mutex};

use anyhow::{anyhow, bail, Result};
use object::{
    elf, Endian, Endianness, File, FileFlags, Object, ObjectSection, ObjectSymbol, Relocation,
    RelocationFlags, RelocationTarget,
};
use rabbitizer::{traits::Register, Abi, DisplayFlags, Instruction, InstructionFlags, IsaExtension, ValuedOperand};

use crate::{
    arch::{ObjArch, ProcessCodeResult},
    diff::{DiffObjConfig, MipsAbi, MipsInstrCategory},
    obj::{ObjIns, ObjInsArg, ObjInsArgValue, ObjReloc, ObjSection},
};

static RABBITIZER_MUTEX: Mutex<()> = Mutex::new(());

pub struct ObjArchMips {
    pub endianness: Endianness,
    pub abi: Abi,
    pub instr_category: IsaExtension,
    pub ri_gp_value: i32,
}

const EF_MIPS_ABI: u32 = 0x0000F000;
const EF_MIPS_MACH: u32 = 0x00FF0000;

const EF_MIPS_MACH_ALLEGREX: u32 = 0x00840000;
const EF_MIPS_MACH_5900: u32 = 0x00920000;

const R_MIPS15_S3: u32 = 119;

impl ObjArchMips {
    pub fn new(object: &File) -> Result<Self> {
        let mut abi = Abi::O32;
        let mut instr_category = IsaExtension::NONE;
        match object.flags() {
            FileFlags::None => {}
            FileFlags::Elf { e_flags, .. } => {
                abi = match e_flags & EF_MIPS_ABI {
                    elf::EF_MIPS_ABI_O32 | elf::EF_MIPS_ABI_O64 => Abi::O32,
                    elf::EF_MIPS_ABI_EABI32 | elf::EF_MIPS_ABI_EABI64 => Abi::N32,
                    _ => {
                        if e_flags & elf::EF_MIPS_ABI2 != 0 {
                            Abi::N32
                        } else {
                            Abi::O32
                        }
                    }
                };
                instr_category = match e_flags & EF_MIPS_MACH {
                    EF_MIPS_MACH_ALLEGREX => IsaExtension::R4000ALLEGREX,
                    EF_MIPS_MACH_5900 => IsaExtension::R5900,
                    _ => IsaExtension::NONE,
                };
            }
            _ => bail!("Unsupported MIPS file flags"),
        }

        // Parse the ri_gp_value stored in .reginfo to be able to correctly
        // calculate R_MIPS_GPREL16 relocations later. The value is stored
        // 0x14 bytes into .reginfo (on 32 bit platforms)
        let ri_gp_value = object
            .section_by_name(".reginfo")
            .and_then(|section| section.data().ok())
            .and_then(|data| data.get(0x14..0x18))
            .and_then(|s| s.try_into().ok())
            .map(|bytes| object.endianness().read_i32_bytes(bytes))
            .unwrap_or(0);

        Ok(Self { endianness: object.endianness(), abi, instr_category, ri_gp_value })
    }
}

impl ObjArch for ObjArchMips {
    fn process_code(
        &self,
        address: u64,
        code: &[u8],
        _section_index: usize,
        relocations: &[ObjReloc],
        line_info: &BTreeMap<u64, u32>,
        config: &DiffObjConfig,
    ) -> Result<ProcessCodeResult> {
        let _guard = RABBITIZER_MUTEX.lock().map_err(|e| anyhow!("Failed to lock mutex: {e}"))?;
        let instruction_flags = InstructionFlags::new().with_abi(match config.mips_abi {
            MipsAbi::Auto => self.abi,
            MipsAbi::O32 => Abi::O32,
            MipsAbi::N32 => Abi::N32,
            MipsAbi::N64 => Abi::N64,
        });
        let instr_category = match config.mips_instr_category {
            MipsInstrCategory::Auto => self.instr_category,
            MipsInstrCategory::Cpu => IsaExtension::NONE,
            MipsInstrCategory::Rsp => IsaExtension::RSP,
            MipsInstrCategory::R3000Gte => IsaExtension::R3000GTE,
            MipsInstrCategory::R4000Allegrex => IsaExtension::R4000ALLEGREX,
            MipsInstrCategory::R5900 => IsaExtension::R5900,
        };
        let display_flags = DisplayFlags::default().with_unknown_instr_comment(false);

        let start_address = address;
        let end_address = address + code.len() as u64;
        let ins_count = code.len() / 4;
        let mut ops = Vec::<u16>::with_capacity(ins_count);
        let mut insts = Vec::<ObjIns>::with_capacity(ins_count);
        let mut cur_addr = start_address as u32;
        for chunk in code.chunks_exact(4) {
            let reloc = relocations.iter().find(|r| (r.address as u32 & !3) == cur_addr);
            let code = self.endianness.read_u32_bytes(chunk.try_into()?);
            let instruction = Instruction::new(code, cur_addr, instruction_flags, instr_category.isa_version(), instr_category);

            let formatted = instruction.display(None, &display_flags).to_string();
            let op = instruction.opcode() as u16;
            ops.push(op);

            let mnemonic = instruction.opcode().name().to_string();
            let mut branch_dest = instruction.get_branch_offset_generic().map(|a| a as u64);

            let operands = instruction.valued_operands_iter();

            let mut args = Vec::with_capacity(5+1); // TODO: hardcoded number
            for (idx, op) in operands.enumerate() {
                if idx > 0 {
                    args.push(ObjInsArg::PlainText(config.separator().into()));
                }

                match op {
                    ValuedOperand::core_immediate(imm) => {
                        if let Some(reloc) = reloc {
                            push_reloc(&mut args, reloc)?;
                        } else {
                            args.push(ObjInsArg::Arg(
                                match imm {
                                    rabbitizer::IU16::Integer(s) => ObjInsArgValue::Signed(s as i64),
                                    rabbitizer::IU16::Unsigned(u) => ObjInsArgValue::Unsigned(u as u64),
                                }
                            ));
                        }
                    }
                    ValuedOperand::core_label(..)
                    | ValuedOperand::core_branch_target_label(..)  => {
                        if let Some(reloc) = reloc {
                            if matches!(&reloc.target_section, Some(s) if s == ".text")
                                && reloc.target.address > start_address
                                && reloc.target.address < end_address
                            {
                                args.push(ObjInsArg::BranchDest(reloc.target.address));
                            } else {
                                push_reloc(&mut args, reloc)?;
                                branch_dest = None;
                            }
                        } else if let Some(branch_dest) = branch_dest {
                            args.push(ObjInsArg::BranchDest(branch_dest));
                        } else {
                            args.push(ObjInsArg::Arg(ObjInsArgValue::Opaque(
                                op.display(&instruction, None, &display_flags).to_string().into(),
                            )));
                        }
                    }
                    ValuedOperand::core_immediate_base(imm, base) => {
                        if let Some(reloc) = reloc {
                            push_reloc(&mut args, reloc)?;
                        } else {
                            args.push(ObjInsArg::Arg(
                                match imm {
                                    rabbitizer::IU16::Integer(s) => ObjInsArgValue::Signed(s as i64),
                                    rabbitizer::IU16::Unsigned(u) => ObjInsArgValue::Unsigned(u as u64),
                                }
                            ));
                        }
                        args.push(ObjInsArg::PlainText("(".into()));
                        args.push(ObjInsArg::Arg(ObjInsArgValue::Opaque(
                            base.either_name(instruction.flags().abi(), display_flags.named_gpr()).into()
                        )));
                        args.push(ObjInsArg::PlainText(")".into()));
                    }
                    // ValuedOperand::r5900_immediate15(..) => match reloc {
                    //     Some(reloc)
                    //         if reloc.flags == RelocationFlags::Elf { r_type: R_MIPS15_S3 } =>
                    //     {
                    //         push_reloc(&mut args, reloc)?;
                    //     }
                    //     _ => {
                    //         args.push(ObjInsArg::Arg(ObjInsArgValue::Opaque(
                    //             op.disassemble(&instruction, None).into(),
                    //         )));
                    //     }
                    // },
                    _ => {
                        args.push(ObjInsArg::Arg(ObjInsArgValue::Opaque(
                            op.display(&instruction, None, &display_flags).to_string().into(),
                        )));
                    }
                }
            }
            let line = line_info.range(..=cur_addr as u64).last().map(|(_, &b)| b);
            insts.push(ObjIns {
                address: cur_addr as u64,
                size: 4,
                op,
                mnemonic,
                args,
                reloc: reloc.cloned(),
                branch_dest,
                line,
                formatted,
                orig: None,
            });
            cur_addr += 4;
        }
        Ok(ProcessCodeResult { ops, insts })
    }

    fn implcit_addend(
        &self,
        file: &File<'_>,
        section: &ObjSection,
        address: u64,
        reloc: &Relocation,
    ) -> Result<i64> {
        let data = section.data[address as usize..address as usize + 4].try_into()?;
        let addend = self.endianness.read_u32_bytes(data);
        Ok(match reloc.flags() {
            RelocationFlags::Elf { r_type: elf::R_MIPS_32 } => addend as i64,
            RelocationFlags::Elf { r_type: elf::R_MIPS_26 } => ((addend & 0x03FFFFFF) << 2) as i64,
            RelocationFlags::Elf { r_type: elf::R_MIPS_HI16 } => {
                ((addend & 0x0000FFFF) << 16) as i32 as i64
            }
            RelocationFlags::Elf {
                r_type: elf::R_MIPS_LO16 | elf::R_MIPS_GOT16 | elf::R_MIPS_CALL16,
            } => (addend & 0x0000FFFF) as i16 as i64,
            RelocationFlags::Elf { r_type: elf::R_MIPS_GPREL16 | elf::R_MIPS_LITERAL } => {
                let RelocationTarget::Symbol(idx) = reloc.target() else {
                    bail!("Unsupported R_MIPS_GPREL16 relocation against a non-symbol");
                };
                let sym = file.symbol_by_index(idx)?;

                // if the symbol we are relocating against is in a local section we need to add
                // the ri_gp_value from .reginfo to the addend.
                if sym.section().index().is_some() {
                    ((addend & 0x0000FFFF) as i16 as i64) + self.ri_gp_value as i64
                } else {
                    (addend & 0x0000FFFF) as i16 as i64
                }
            }
            RelocationFlags::Elf { r_type: elf::R_MIPS_PC16 } => 0, // PC-relative relocation
            RelocationFlags::Elf { r_type: R_MIPS15_S3 } => ((addend & 0x001FFFC0) >> 3) as i64,
            flags => bail!("Unsupported MIPS implicit relocation {flags:?}"),
        })
    }

    fn display_reloc(&self, flags: RelocationFlags) -> Cow<'static, str> {
        match flags {
            RelocationFlags::Elf { r_type } => match r_type {
                elf::R_MIPS_32 => Cow::Borrowed("R_MIPS_32"),
                elf::R_MIPS_26 => Cow::Borrowed("R_MIPS_26"),
                elf::R_MIPS_HI16 => Cow::Borrowed("R_MIPS_HI16"),
                elf::R_MIPS_LO16 => Cow::Borrowed("R_MIPS_LO16"),
                elf::R_MIPS_GPREL16 => Cow::Borrowed("R_MIPS_GPREL16"),
                elf::R_MIPS_LITERAL => Cow::Borrowed("R_MIPS_LITERAL"),
                elf::R_MIPS_GOT16 => Cow::Borrowed("R_MIPS_GOT16"),
                elf::R_MIPS_PC16 => Cow::Borrowed("R_MIPS_PC16"),
                elf::R_MIPS_CALL16 => Cow::Borrowed("R_MIPS_CALL16"),
                R_MIPS15_S3 => Cow::Borrowed("R_MIPS15_S3"),
                _ => Cow::Owned(format!("<{flags:?}>")),
            },
            _ => Cow::Owned(format!("<{flags:?}>")),
        }
    }
}

fn push_reloc(args: &mut Vec<ObjInsArg>, reloc: &ObjReloc) -> Result<()> {
    match reloc.flags {
        RelocationFlags::Elf { r_type } => match r_type {
            elf::R_MIPS_HI16 => {
                args.push(ObjInsArg::PlainText("%hi(".into()));
                args.push(ObjInsArg::Reloc);
                args.push(ObjInsArg::PlainText(")".into()));
            }
            elf::R_MIPS_LO16 => {
                args.push(ObjInsArg::PlainText("%lo(".into()));
                args.push(ObjInsArg::Reloc);
                args.push(ObjInsArg::PlainText(")".into()));
            }
            elf::R_MIPS_GOT16 => {
                args.push(ObjInsArg::PlainText("%got(".into()));
                args.push(ObjInsArg::Reloc);
                args.push(ObjInsArg::PlainText(")".into()));
            }
            elf::R_MIPS_CALL16 => {
                args.push(ObjInsArg::PlainText("%call16(".into()));
                args.push(ObjInsArg::Reloc);
                args.push(ObjInsArg::PlainText(")".into()));
            }
            elf::R_MIPS_GPREL16 => {
                args.push(ObjInsArg::PlainText("%gp_rel(".into()));
                args.push(ObjInsArg::Reloc);
                args.push(ObjInsArg::PlainText(")".into()));
            }
            elf::R_MIPS_32
            | elf::R_MIPS_26
            | elf::R_MIPS_LITERAL
            | elf::R_MIPS_PC16
            | R_MIPS15_S3 => {
                args.push(ObjInsArg::Reloc);
            }
            _ => bail!("Unsupported ELF MIPS relocation type {r_type}"),
        },
        flags => panic!("Unsupported MIPS relocation flags {flags:?}"),
    }
    Ok(())
}
