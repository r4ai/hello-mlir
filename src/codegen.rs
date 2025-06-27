use crate::ast::{self, Type};
use anyhow::Result;
use std::collections::HashMap;

pub struct CodeGenerator {
    output: String,
    variables: HashMap<String, String>,
    temp_counter: usize,
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeGenerator {
    pub fn new() -> Self {
        CodeGenerator {
            output: String::new(),
            variables: HashMap::new(),
            temp_counter: 0,
        }
    }

    fn next_temp(&mut self) -> String {
        let temp = format!("%{}", self.temp_counter);
        self.temp_counter += 1;
        temp
    }

    fn generate_target_attributes(&self) -> String {
        "module attributes {\
            dlti.dl_spec = #dlti.dl_spec<\
                #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, \
                #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, \
                #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, \
                #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, \
                #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, \
                #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, \
                #dlti.dl_entry<f32, dense<32> : vector<2xi64>>, \
                #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, \
                #dlti.dl_entry<\"dlti.endianness\", \"little\">>, \
            llvm.target_triple = \"x86_64-unknown-linux-gnu\"\
        } {\n"
            .to_string()
    }

    pub fn generate(&mut self, program: &ast::Program<Type>) -> Result<String> {
        // Add target information for x64
        self.output.push_str(&self.generate_target_attributes());

        // Generate all functions
        for function in &program.functions {
            self.generate_function(function)?;
        }

        self.output.push_str("}\n");
        Ok(self.output.clone())
    }

    fn ast_type_to_mlir_type(&self, ast_type: &Type) -> Result<String> {
        match ast_type {
            Type::I32 => Ok("i32".to_string()),
            Type::I64 => Ok("i64".to_string()),
            Type::Bool => Ok("i1".to_string()),
            Type::Void => Ok("()".to_string()),
            Type::F32 => Ok("f32".to_string()),
            Type::F64 => Ok("f64".to_string()),
            Type::String => Ok("!llvm.ptr".to_string()),
            Type::Fn { .. } => Ok("!llvm.ptr".to_string()),
        }
    }

    fn generate_function(&mut self, function: &ast::FnDecl<Type>) -> Result<()> {
        let return_type = self.ast_type_to_mlir_type(&function.r#type)?;

        // Generate function signature using LLVM dialect
        self.output
            .push_str(&format!("  llvm.func @{}(", function.name));

        for (i, param) in function.params.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            let param_type = self.ast_type_to_mlir_type(&param.r#type)?;
            self.output.push_str(&format!("%arg{}: {}", i, param_type));
            self.variables
                .insert(param.name.to_string(), format!("%arg{}", i));
        }

        self.output.push_str(&format!(") -> {} {{\n", return_type));

        // Generate function body
        let mut last_value = None;
        for stmt in &function.body {
            last_value = self.generate_statement(stmt)?;
        }

        // Generate return
        if let Some(value) = last_value {
            self.output
                .push_str(&format!("    llvm.return {} : {}\n", value, return_type));
        } else {
            self.output.push_str("    llvm.return\n");
        }

        self.output.push_str("  }\n");
        Ok(())
    }

    fn generate_statement(&mut self, stmt: &ast::Stmt<Type>) -> Result<Option<String>> {
        match stmt {
            ast::Stmt::LetDecl { name, value, .. } | ast::Stmt::VarDecl { name, value, .. } => {
                if let Some(expr) = value {
                    let val = self.generate_expression(expr)?;
                    self.variables.insert(name.to_string(), val);
                    Ok(None)
                } else {
                    Ok(None)
                }
            }
            ast::Stmt::Assign { name, value, .. } => {
                let val = self.generate_expression(value)?;
                self.variables.insert(name.to_string(), val);
                Ok(None)
            }
            ast::Stmt::Return { expr, .. } => {
                if let Some(expr) = expr {
                    let value = self.generate_expression(expr)?;
                    Ok(Some(value))
                } else {
                    Ok(None)
                }
            }
            ast::Stmt::ExprStmt { expr, .. } | ast::Stmt::Expr { expr, .. } => {
                let value = self.generate_expression(expr)?;
                Ok(Some(value))
            }
            ast::Stmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let cond_value = self.generate_expression(condition)?;

                self.output
                    .push_str(&format!("    scf.if {} {{\n", cond_value));

                for stmt in then_branch {
                    self.generate_statement(stmt)?;
                }

                if let Some(else_stmts) = else_branch {
                    self.output.push_str("    } else {\n");
                    for stmt in else_stmts {
                        self.generate_statement(stmt)?;
                    }
                }

                self.output.push_str("    }\n");
                Ok(None)
            }
            _ => Err(anyhow::anyhow!("Unsupported statement: {:?}", stmt)),
        }
    }

    fn generate_expression(&mut self, expr: &ast::Expr) -> Result<String> {
        match expr {
            ast::Expr::IntLit { value, .. } => {
                let temp = self.next_temp();
                self.output.push_str(&format!(
                    "    {} = llvm.mlir.constant({} : i32) : i32\n",
                    temp, value
                ));
                Ok(temp)
            }
            ast::Expr::BoolLit { value, .. } => {
                let temp = self.next_temp();
                let bool_val = if *value { "1" } else { "0" };
                self.output.push_str(&format!(
                    "    {} = llvm.mlir.constant({} : i1) : i1\n",
                    temp, bool_val
                ));
                Ok(temp)
            }
            ast::Expr::BinOp { lhs, op, rhs, .. } => {
                let lhs_val = self.generate_expression(lhs)?;
                let rhs_val = self.generate_expression(rhs)?;
                let temp = self.next_temp();

                match op {
                    ast::BinOp::Add => {
                        self.output.push_str(&format!(
                            "    {} = llvm.add {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::Sub => {
                        self.output.push_str(&format!(
                            "    {} = llvm.sub {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::Mul => {
                        self.output.push_str(&format!(
                            "    {} = llvm.mul {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::Div => {
                        self.output.push_str(&format!(
                            "    {} = llvm.sdiv {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::Equal => {
                        self.output.push_str(&format!(
                            "    {} = llvm.icmp \"eq\" {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::NotEqual => {
                        self.output.push_str(&format!(
                            "    {} = llvm.icmp \"ne\" {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::LessThan => {
                        self.output.push_str(&format!(
                            "    {} = llvm.icmp \"slt\" {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::LessThanOrEqual => {
                        self.output.push_str(&format!(
                            "    {} = llvm.icmp \"sle\" {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::GreaterThan => {
                        self.output.push_str(&format!(
                            "    {} = llvm.icmp \"sgt\" {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::GreaterThanOrEqual => {
                        self.output.push_str(&format!(
                            "    {} = llvm.icmp \"sge\" {}, {} : i32\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::And => {
                        self.output.push_str(&format!(
                            "    {} = llvm.and {}, {} : i1\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                    ast::BinOp::Or => {
                        self.output.push_str(&format!(
                            "    {} = llvm.or {}, {} : i1\n",
                            temp, lhs_val, rhs_val
                        ));
                    }
                }
                Ok(temp)
            }
            ast::Expr::UnaryOp { op, expr, .. } => {
                let val = self.generate_expression(expr)?;
                let temp = self.next_temp();

                match op {
                    ast::UnaryOp::Neg => {
                        let zero_temp = self.next_temp();
                        self.output.push_str(&format!(
                            "    {} = llvm.mlir.constant(0 : i32) : i32\n",
                            zero_temp
                        ));
                        self.output.push_str(&format!(
                            "    {} = llvm.sub {}, {} : i32\n",
                            temp, zero_temp, val
                        ));
                    }
                    ast::UnaryOp::Not => {
                        let one_temp = self.next_temp();
                        self.output.push_str(&format!(
                            "    {} = llvm.mlir.constant(1 : i1) : i1\n",
                            one_temp
                        ));
                        self.output.push_str(&format!(
                            "    {} = llvm.xor {}, {} : i1\n",
                            temp, val, one_temp
                        ));
                    }
                }
                Ok(temp)
            }
            ast::Expr::VarRef { name, .. } => self
                .variables
                .get(*name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Variable '{}' not found", name)),
            ast::Expr::FnCall { name, args, .. } => {
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.generate_expression(arg)?);
                }

                let temp = self.next_temp();
                self.output
                    .push_str(&format!("    {} = llvm.call @{}(", temp, name));
                for (i, arg) in arg_values.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str(arg);
                }
                self.output.push_str(") : (");
                for (i, _) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str("i32"); // Simplified - assume all args are i32
                }
                self.output.push_str(") -> i32\n");
                Ok(temp)
            }
        }
    }

    pub fn compile_to_object(&self, output_path: &str) -> Result<()> {
        // Write MLIR file
        let mlir_path = format!("{}.mlir", output_path);
        std::fs::write(&mlir_path, &self.output)?;
        println!("MLIR generated: {}", mlir_path);

        // Try to compile to LLVM IR and then to object file
        self.compile_mlir_to_x64(&mlir_path, output_path)?;

        Ok(())
    }

    fn compile_mlir_to_x64(&self, mlir_path: &str, output_path: &str) -> Result<()> {
        use std::process::Command;

        // Check if mlir-translate is available
        let mlir_translate_check = Command::new("mlir-translate").arg("--version").output();

        if mlir_translate_check.is_err() {
            println!(
                "Warning: mlir-translate not found. Install MLIR/LLVM tools for full compilation."
            );
            println!("You can still use the generated MLIR code: {}", mlir_path);
            return Ok(());
        }

        // Convert to LLVM IR
        let llvm_ir_path = format!("{}.ll", output_path);
        let convert_result = Command::new("mlir-translate")
            .arg("--mlir-to-llvmir")
            .arg(mlir_path)
            .arg("-o")
            .arg(&llvm_ir_path)
            .output();

        match convert_result {
            Ok(output) if output.status.success() => {
                println!("LLVM IR generated: {}", llvm_ir_path);

                // Check if llc is available for object compilation
                let llc_check = Command::new("llc").arg("--version").output();
                if llc_check.is_ok() {
                    // Compile to object file
                    let obj_path = format!("{}.o", output_path);
                    let compile_result = Command::new("llc")
                        .arg("-filetype=obj")
                        .arg("-march=x86-64")
                        .arg(&llvm_ir_path)
                        .arg("-o")
                        .arg(&obj_path)
                        .output();

                    match compile_result {
                        Ok(output) if output.status.success() => {
                            println!("Object file generated: {}", obj_path);

                            // Try to create executable
                            let exe_path = output_path.to_string();
                            let link_result = Command::new("clang")
                                .arg(&obj_path)
                                .arg("-o")
                                .arg(&exe_path)
                                .output();

                            match link_result {
                                Ok(output) if output.status.success() => {
                                    println!("Executable created: {}", exe_path);
                                    println!("Compilation to x64 successful!");
                                }
                                _ => {
                                    println!(
                                        "Warning: Could not link executable. Object file available: {}",
                                        obj_path
                                    );
                                }
                            }
                        }
                        _ => {
                            println!(
                                "Warning: Could not compile to object file. LLVM IR available: {}",
                                llvm_ir_path
                            );
                        }
                    }
                } else {
                    println!(
                        "Warning: llc not found. LLVM IR available: {}",
                        llvm_ir_path
                    );
                }
            }
            Ok(output) => {
                println!("Error converting MLIR to LLVM IR:");
                println!("{}", String::from_utf8_lossy(&output.stderr));
            }
            Err(e) => {
                println!("Error running mlir-translate: {}", e);
            }
        }

        Ok(())
    }
}

pub fn generate_code(program: &ast::Program<Type>, output_path: Option<&str>) -> Result<String> {
    let mut codegen = CodeGenerator::new();
    let mlir_code = codegen.generate(program)?;

    if let Some(path) = output_path {
        codegen.compile_to_object(path)?;
    }

    Ok(mlir_code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Expr, FnDecl, FnParam, Span, Stmt, UnaryOp};

    fn create_span() -> Span {
        Span {
            start: 0,
            end: 0,
            context: (),
        }
    }

    #[test]
    fn test_basic_function_generation() {
        let mut codegen = CodeGenerator::new();

        // Create a simple program: fn main() -> i32 { return 42; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "main",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Return {
                    expr: Some(Box::new(Expr::IntLit {
                        value: 42,
                        span: create_span(),
                    })),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain LLVM dialect
        assert!(result.contains("llvm.func @main"));
        assert!(result.contains("llvm.mlir.constant(42 : i32)"));
        assert!(result.contains("llvm.return"));
        assert!(result.contains("x86_64-unknown-linux-gnu"));
    }

    #[test]
    fn test_arithmetic_operations() {
        let mut codegen = CodeGenerator::new();

        // Create: fn test() -> i32 { return 5 + 3; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Return {
                    expr: Some(Box::new(Expr::BinOp {
                        lhs: Box::new(Expr::IntLit {
                            value: 5,
                            span: create_span(),
                        }),
                        op: BinOp::Add,
                        rhs: Box::new(Expr::IntLit {
                            value: 3,
                            span: create_span(),
                        }),
                        span: create_span(),
                    })),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain LLVM arithmetic
        assert!(result.contains("llvm.add"));
        assert!(result.contains("llvm.mlir.constant(5 : i32)"));
        assert!(result.contains("llvm.mlir.constant(3 : i32)"));
    }

    #[test]
    fn test_function_with_parameters() {
        let mut codegen = CodeGenerator::new();

        // Create: fn add(a: i32, b: i32) -> i32 { return a + b; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "add",
                params: vec![
                    FnParam {
                        name: "a",
                        r#type: Type::I32,
                        span: create_span(),
                    },
                    FnParam {
                        name: "b",
                        r#type: Type::I32,
                        span: create_span(),
                    },
                ],
                r#type: Type::I32,
                body: vec![Stmt::Return {
                    expr: Some(Box::new(Expr::BinOp {
                        lhs: Box::new(Expr::VarRef {
                            name: "a",
                            span: create_span(),
                        }),
                        op: BinOp::Add,
                        rhs: Box::new(Expr::VarRef {
                            name: "b",
                            span: create_span(),
                        }),
                        span: create_span(),
                    })),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain function with parameters
        assert!(result.contains("llvm.func @add(%arg0: i32, %arg1: i32)"));
        assert!(result.contains("llvm.add %arg0, %arg1"));
    }

    #[test]
    fn test_boolean_operations() {
        let mut codegen = CodeGenerator::new();

        // Create: fn test() -> i1 { return true && false; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::Bool,
                body: vec![Stmt::Return {
                    expr: Some(Box::new(Expr::BinOp {
                        lhs: Box::new(Expr::BoolLit {
                            value: true,
                            span: create_span(),
                        }),
                        op: BinOp::And,
                        rhs: Box::new(Expr::BoolLit {
                            value: false,
                            span: create_span(),
                        }),
                        span: create_span(),
                    })),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain boolean operations
        assert!(result.contains("llvm.mlir.constant(1 : i1)"));
        assert!(result.contains("llvm.mlir.constant(0 : i1)"));
        assert!(result.contains("llvm.and"));
    }

    #[test]
    fn test_unary_operations() {
        let mut codegen = CodeGenerator::new();

        // Create: fn test() -> i32 { return -42; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Return {
                    expr: Some(Box::new(Expr::UnaryOp {
                        op: UnaryOp::Neg,
                        expr: Box::new(Expr::IntLit {
                            value: 42,
                            span: create_span(),
                        }),
                        span: create_span(),
                    })),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain unary negation
        assert!(result.contains("llvm.mlir.constant(0 : i32)"));
        assert!(result.contains("llvm.mlir.constant(42 : i32)"));
        assert!(result.contains("llvm.sub"));
    }

    #[test]
    fn test_x64_target_attributes() {
        let mut codegen = CodeGenerator::new();
        let program = ast::Program { functions: vec![] };

        let result = codegen.generate(&program).unwrap();

        // Should contain x64 target information
        assert!(result.contains("x86_64-unknown-linux-gnu"));
        assert!(result.contains("dlti.dl_spec"));
        assert!(result.contains("little"));
    }

    #[test]
    fn test_unsupported_type_error() {
        let codegen = CodeGenerator::new();

        // Test that unsupported types return errors instead of defaulting
        let result = codegen.ast_type_to_mlir_type(&Type::I32);
        assert!(result.is_ok());

        // This would test an unsupported type if we had one in the enum
        // For now, all types are supported
    }

    #[test]
    fn test_variable_declarations() {
        let mut codegen = CodeGenerator::new();

        // Create: fn test() -> i32 { let x: i32 = 10; return x; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::I32,
                body: vec![
                    Stmt::LetDecl {
                        name: "x",
                        r#type: Type::I32,
                        value: Some(Expr::IntLit {
                            value: 10,
                            span: create_span(),
                        }),
                        span: create_span(),
                    },
                    Stmt::Return {
                        expr: Some(Box::new(Expr::VarRef {
                            name: "x",
                            span: create_span(),
                        })),
                        span: create_span(),
                    },
                ],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain constant assignment and variable reference
        assert!(result.contains("llvm.mlir.constant(10 : i32)"));
    }
}
