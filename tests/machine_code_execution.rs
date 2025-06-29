use hello_mlir::{codegen, parser, typechecker};
use indoc::indoc;
use std::process::Command;
use tempfile::TempDir;

/// Create a test program that compiles, type-checks, and generates MLIR
fn create_test_program(source: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Parse the source code
    let program = match parser::parse(source).into_result() {
        Ok(program) => program,
        Err(errors) => {
            return Err(format!("Parse errors: {:?}", errors).into());
        }
    };

    // Type check the program
    let typed_program = typechecker::typecheck(&program, "test")
        .map_err(|err| format!("Type check error: {:?}", err))?;

    // Generate MLIR code
    let mlir_code = codegen::generate_code(&typed_program, None)?;

    Ok(mlir_code)
}

/// Compile source code to executable and run it
fn compile_and_run_source(source: &str) -> Result<i32, Box<dyn std::error::Error>> {
    if !check_mlir_tools_available() {
        return Err("MLIR tools not available. Please install LLVM/MLIR toolchain.".into());
    }

    // Parse the source code
    let program = match parser::parse(source).into_result() {
        Ok(program) => program,
        Err(errors) => {
            return Err(format!("Parse errors: {:?}", errors).into());
        }
    };

    // Type check the program
    let typed_program = typechecker::typecheck(&program, "test")
        .map_err(|err| format!("Type check error: {:?}", err))?;

    let temp_dir = TempDir::new()?;
    let exe_path = temp_dir.path().join("test");

    // Generate executable using codegen with LLVM conversion
    codegen::generate_code(&typed_program, Some(exe_path.to_str().unwrap()))?;

    // Run the executable
    let output = Command::new(&exe_path).output()?;

    // Get exit code - note that non-zero exit codes are normal for our programs
    // They return the computed value as the exit code
    let exit_code = output.status.code().unwrap_or(-1);

    // Only consider it a failure if the program crashed (exit code < 0) or had stderr output
    if exit_code < 0 || !output.stderr.is_empty() {
        return Err(format!(
            "Program execution failed: stdout: {}, stderr: {}, status: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
            output.status
        )
        .into());
    }

    Ok(exit_code)
}

/// Check if MLIR tools are available
fn check_mlir_tools_available() -> bool {
    Command::new("mlir-translate")
        .arg("--version")
        .output()
        .is_ok()
        && Command::new("llc").arg("--version").output().is_ok()
        && Command::new("clang").arg("--version").output().is_ok()
}

#[test]
fn test_simple_return_value() {
    let source = r#"
        fn main() -> i32 {
            42
        }
    "#;

    let mlir_code = create_test_program(source).expect("Failed to create test program");

    // Test that MLIR is generated correctly
    assert!(mlir_code.contains("func.func @main"));
    assert!(mlir_code.contains("arith.constant 42"));
    assert!(mlir_code.contains("return"));

    // Test actual execution now that LLVM dialect conversion is implemented
    let exit_code = compile_and_run_source(source).expect("Failed to compile and run");
    assert_eq!(exit_code, 42, "Program should return 42");
    println!("✅ Execution test passed: exit code = {}", exit_code);
}

#[test]
fn test_arithmetic_operations() {
    let source = r#"
        fn main() -> i32 {
            10 + 5 * 2
        }
    "#;

    let mlir_code = create_test_program(source).expect("Failed to create test program");

    // Test that MLIR contains arithmetic operations
    assert!(mlir_code.contains("arith.constant 10"));
    assert!(mlir_code.contains("arith.constant 5"));
    assert!(mlir_code.contains("arith.constant 2"));
    assert!(mlir_code.contains("arith.muli"));
    assert!(mlir_code.contains("arith.addi"));

    // Test actual execution
    let exit_code = compile_and_run_source(source).expect("Failed to compile and run");
    assert_eq!(exit_code, 20, "Program should return 20 (10 + 5 * 2)");
    println!("✅ Arithmetic test passed: exit code = {}", exit_code);
}

#[test]
fn test_function_with_parameters() {
    let source = r#"
        fn add(a: i32, b: i32) -> i32 {
            a + b
        }
        
        fn main() -> i32 {
            add(15, 27)
        }
    "#;

    let mlir_code = create_test_program(source).expect("Failed to create test program");

    // Test that MLIR contains function definitions and calls
    assert!(mlir_code.contains("func.func @add"));
    assert!(mlir_code.contains("func.func @main"));
    assert!(mlir_code.contains("call @add"));

    let exit_code = compile_and_run_source(source).expect("Failed to compile and run");
    assert_eq!(exit_code, 42, "Program should return 42 (15 + 27)");
    println!("✅ Function call test passed: exit code = {}", exit_code);
}

#[test]
fn test_variable_declarations() {
    let source = r#"
        fn main() -> i32 {
            let x: i32 = 10;
            let y: i32 = 20;
            x + y
        }
    "#;

    let mlir_code = create_test_program(source).expect("Failed to create test program");

    // Test that MLIR is generated
    assert!(mlir_code.contains("func.func @main"));
    assert!(mlir_code.contains("arith.constant"));

    // Test actual execution
    let exit_code = compile_and_run_source(source).expect("Failed to compile and run");
    assert_eq!(exit_code, 30, "Program should return 30 (10 + 20)");
    println!("✅ Variable test passed: exit code = {}", exit_code);
}

#[test]
fn test_boolean_operations() {
    let source = r#"
        fn main() -> i32 {
            let x: bool = true;
            let y: bool = false;
            if x { 1 } else { 0 }
        }
    "#;

    let mlir_code = create_test_program(source).expect("Failed to create test program");

    // Test that MLIR contains boolean constants
    assert!(mlir_code.contains("func.func @main"));
    assert!(mlir_code.contains("arith.constant"));
    assert!(mlir_code.contains("scf.if"));
    assert!(mlir_code.contains("scf.yield"));
    assert!(mlir_code.contains("return"));
}

#[test]
fn test_nested_if_expression() {
    let source = indoc! {"
        fn test(x: i32) -> i32 {
            if x > 0 {
                if x < 10 {
                    x + 1
                } else {
                    x - 1
                }
            } else {
                0
            }
        }

        fn main() -> i32 {
            test(3)
        }
    "};

    let exit_code = compile_and_run_source(source).expect("Failed to compile and run");
    assert_eq!(exit_code, 4, "Program should return 4 (= 3 + 1)");
}

#[test]
fn test_mlir_verification() {
    // Test that generated MLIR passes verification
    let source = r#"
        fn fibonacci(n: i32) -> i32 {
            n + 1
        }
        
        fn main() -> i32 {
            fibonacci(5)
        }
    "#;

    let mlir_code = create_test_program(source).expect("Failed to create test program");

    // Write MLIR to temporary file and verify it
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mlir_path = temp_dir.path().join("test.mlir");
    std::fs::write(&mlir_path, &mlir_code).expect("Failed to write MLIR file");

    // Try to parse with mlir-opt if available
    let output = Command::new("mlir-opt")
        .arg("--verify-diagnostics")
        .arg(&mlir_path)
        .output()
        .expect("Failed to run mlir-opt");
    if !output.status.success() {
        panic!(
            "MLIR verification failed:\nMLIR Code:\n{}\nError:\n{}",
            mlir_code,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
