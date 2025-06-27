mod ast;
mod codegen;
mod parser;
mod token;
mod typechecker;

use anyhow::Result;
use ariadne::{Report, ReportKind};
use clap::Parser;
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CompileMode {
    /// Emit AST
    Ast,

    /// Emit Machine Code
    MachineCode,
}

/// A simple integer-only compiler
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Input file to compile
    input: PathBuf,

    /// Output file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Mode of compilation
    #[arg(short, long, default_value_t = CompileMode::MachineCode, ignore_case = true, value_enum)]
    mode: CompileMode,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Read the input file
    let file_id = args.input.to_string_lossy().to_string();
    let input = fs::read_to_string(&file_id)?;

    // Parse the input
    let program = match parser::parse(&input).into_result() {
        Ok(program) => program,
        Err(errors) => {
            for err in errors {
                Report::build(ReportKind::Error, ((), err.span().into_range()))
                    .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
                    .with_code(3)
                    .with_message(err.to_string())
                    .with_label(
                        ariadne::Label::new(((), err.span().into_range()))
                            .with_message(err.reason().to_string())
                            .with_color(ariadne::Color::Red),
                    )
                    .finish()
                    .eprint(ariadne::Source::from(&input))
                    .unwrap();
            }
            return Err(anyhow::anyhow!("Failed to parse input"));
        }
    };

    // Handle the different compilation modes
    match args.mode {
        CompileMode::Ast => {
            let ast_yaml = serde_yaml::to_string(&program).unwrap();
            println!("{ast_yaml}");
        }
        CompileMode::MachineCode => {
            let typed_program = typechecker::typecheck_and_transform(&program, &file_id).map_err(|err| {
                Report::build(ReportKind::Error, ((), err.span.to_range()))
                    .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
                    .with_code(3)
                    .with_message(err.to_string())
                    .with_label(
                        ariadne::Label::new(((), err.span.to_range()))
                            .with_message(err.to_string())
                            .with_color(ariadne::Color::Red),
                    )
                    .finish()
                    .eprint(ariadne::Source::from(&input))
                    .unwrap();
                anyhow::anyhow!("Type checking failed")
            })?;

            let output_path = args.output.as_ref().map(|p| p.to_string_lossy().to_string());
            let mlir_code = codegen::generate_code(&typed_program, output_path.as_deref())?;
            
            if args.output.is_none() {
                println!("{}", mlir_code);
            }
        }
    }

    Ok(())
}
