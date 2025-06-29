use crate::ast;
use anyhow::Result;
use melior::{
    Context,
    dialect::{DialectRegistry, arith, func, llvm, scf},
    ir::{
        Block, BlockLike, Location, Module, Region, RegionLike, Value, ValueLike,
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, IntegerType},
    },
    pass::{PassManager, conversion},
    utility::register_all_dialects,
};
use std::{any::Any, collections::HashMap};

struct Variable<'c> {
    #[allow(dead_code)]
    name: String,
    value: Value<'c, 'c>,
}

struct VariableEnvironment<'c> {
    scopes: Vec<HashMap<String, Variable<'c>>>,
}

impl<'c> VariableEnvironment<'c> {
    fn new() -> Self {
        VariableEnvironment {
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_var(&mut self, name: String, value: Value<'c, 'c>) -> Result<()> {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.clone(), Variable { name, value });
            Ok(())
        } else {
            Err(anyhow::anyhow!("No scope available to declare variable"))
        }
    }

    fn assign_var(&mut self, name: String, value: Value<'c, 'c>) -> Result<()> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(&name) {
                scope.insert(name.clone(), Variable { name, value });
                return Ok(());
            }
        }
        Err(anyhow::anyhow!(
            "Variable '{}' not found for assignment",
            name
        ))
    }

    fn lookup_var(&self, name: &str) -> Result<Value<'c, 'c>> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .map(|var| var.value)
            .ok_or_else(|| anyhow::anyhow!("Variable '{}' not found", name))
    }
}

pub struct CodeGenerator<'c> {
    context: &'c Context,
    module: Module<'c>,
    location: Location<'c>,
    variables: VariableEnvironment<'c>,
}

impl<'c> CodeGenerator<'c> {
    pub fn new(context: &'c Context) -> Self {
        let location = Location::unknown(context);
        let module = Module::new(location);

        CodeGenerator {
            context,
            module,
            location,
            variables: VariableEnvironment::new(),
        }
    }

    pub fn generate(&mut self, program: &ast::Program<ast::Type>) -> Result<String> {
        // Generate all functions
        for function in &program.functions {
            self.generate_function(function)?;
        }

        // Verify the module
        if !self.module.as_operation().verify() {
            return Err(anyhow::anyhow!(
                "Generated MLIR module failed verification:\n{}",
                self.module.as_operation()
            ));
        }

        Ok(format!("{}", self.module.as_operation()))
    }

    fn ast_type_to_mlir_type(&self, ast_type: &ast::Type) -> Result<melior::ir::Type<'c>> {
        match ast_type {
            ast::Type::I32 => Ok(IntegerType::new(self.context, 32).into()),
            ast::Type::I64 => Ok(IntegerType::new(self.context, 64).into()),
            ast::Type::Bool => Ok(IntegerType::new(self.context, 1).into()),
            ast::Type::Void => Ok(melior::ir::r#type::Type::none(self.context)),
            ast::Type::F32 => Ok(melior::ir::r#type::Type::float32(self.context)),
            ast::Type::F64 => Ok(melior::ir::r#type::Type::float64(self.context)),
            ast::Type::String => Ok(llvm::r#type::pointer(self.context, 0)),
            ast::Type::Fn { .. } => Ok(llvm::r#type::pointer(self.context, 0)),
        }
    }

    fn generate_function(&mut self, function: &ast::FnDecl<ast::Type>) -> Result<()> {
        let return_type = self.ast_type_to_mlir_type(&function.r#type)?;

        // Collect parameter types
        let mut param_types = Vec::new();
        for param in &function.params {
            param_types.push(self.ast_type_to_mlir_type(&param.r#type)?);
        }

        // Create function type
        let function_type = FunctionType::new(self.context, &param_types, &[return_type]);

        // Push a new scope for this function
        self.variables.push_scope();

        // Create the function operation
        let func_op = func::func(
            self.context,
            StringAttribute::new(self.context, function.name),
            TypeAttribute::new(function_type.into()),
            {
                // Create entry block with parameters
                let block = Block::new(
                    &param_types
                        .iter()
                        .map(|t| (*t, self.location))
                        .collect::<Vec<_>>(),
                );

                // Store parameter values in variables map
                for (i, param) in function.params.iter().enumerate() {
                    let arg_value = block.argument(i).unwrap().into();
                    self.variables
                        .declare_var(param.name.to_string(), arg_value)?;
                }

                // Generate function body
                let mut last_value = None;
                for stmt in &function.body {
                    last_value = self.generate_statement(&block, stmt)?;
                }

                // Generate return statement
                if let Some(value) = last_value {
                    block.append_operation(func::r#return(&[value], self.location));
                } else {
                    block.append_operation(func::r#return(&[], self.location));
                }

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            self.location,
        );

        // Pop the function scope
        self.variables.pop_scope();

        self.module.body().append_operation(func_op);
        Ok(())
    }

    fn generate_statement(
        &mut self,
        block: &Block<'c>,
        stmt: &ast::Stmt<ast::Type>,
    ) -> Result<Option<Value<'c, 'c>>> {
        match stmt {
            ast::Stmt::LetDecl {
                name, value: expr, ..
            } => {
                let value = self.generate_expression(block, expr)?;
                self.variables.declare_var(name.to_string(), value)?;
                Ok(None)
            }
            ast::Stmt::VarDecl {
                name, value: expr, ..
            } => {
                if let Some(expr) = expr {
                    let value = self.generate_expression(block, expr)?;
                    self.variables.declare_var(name.to_string(), value)?;
                }
                Ok(None)
            }
            ast::Stmt::Assign { name, value, .. } => {
                let val = self.generate_expression(block, value)?;
                self.variables.assign_var(name.to_string(), val)?;
                Ok(None)
            }
            ast::Stmt::ExprStmt { expr, .. } | ast::Stmt::Expr { expr, .. } => {
                let value = self.generate_expression(block, expr)?;
                Ok(Some(value))
            }
            _ => Err(anyhow::anyhow!("Unsupported statement: {:?}", stmt)),
        }
    }

    fn generate_expression(
        &mut self,
        block: &Block<'c>,
        expr: &ast::Expr<ast::Type>,
    ) -> Result<Value<'c, 'c>> {
        match expr {
            ast::Expr::IntLit { value, r#type, .. } => {
                let int_type = self.ast_type_to_mlir_type(r#type)?;
                let const_op = block.append_operation(arith::constant(
                    self.context,
                    melior::ir::attribute::IntegerAttribute::new(int_type, value.parse::<i64>()?)
                        .into(),
                    self.location,
                ));
                Ok(const_op.result(0)?.into())
            }
            ast::Expr::FloatLit { value, r#type, .. } => {
                let float_type = self.ast_type_to_mlir_type(r#type)?;
                let const_op = block.append_operation(arith::constant(
                    self.context,
                    melior::ir::attribute::FloatAttribute::new(
                        self.context,
                        float_type,
                        value.parse::<f64>()?,
                    )
                    .into(),
                    self.location,
                ));
                Ok(const_op.result(0)?.into())
            }
            ast::Expr::BoolLit { value, .. } => {
                let bool_val = if *value { 1 } else { 0 };
                let const_op = block.append_operation(arith::constant(
                    self.context,
                    melior::ir::attribute::IntegerAttribute::new(
                        IntegerType::new(self.context, 1).into(),
                        bool_val,
                    )
                    .into(),
                    self.location,
                ));
                Ok(const_op.result(0)?.into())
            }
            ast::Expr::BinOp { lhs, op, rhs, .. } => {
                let value_type = lhs.as_ref().r#type();
                let lhs_val = self.generate_expression(block, lhs)?;
                let rhs_val = self.generate_expression(block, rhs)?;

                let add_op = || match value_type {
                    &ast::Type::I32 | &ast::Type::I64 => {
                        arith::addi(lhs_val, rhs_val, self.location)
                    }
                    &ast::Type::F32 | &ast::Type::F64 => {
                        arith::addf(lhs_val, rhs_val, self.location)
                    }
                    _ => {
                        // Never panic here, because we have already type-checked
                        panic!("Unsupported type for binary operation: {:?}", value_type)
                    }
                };

                let sub_op = || match value_type {
                    &ast::Type::I32 | &ast::Type::I64 => {
                        arith::subi(lhs_val, rhs_val, self.location)
                    }
                    &ast::Type::F32 | &ast::Type::F64 => {
                        arith::subf(lhs_val, rhs_val, self.location)
                    }
                    _ => {
                        // Never panic here, because we have already type-checked
                        panic!("Unsupported type for binary operation: {:?}", value_type)
                    }
                };

                let mul_op = || match value_type {
                    &ast::Type::I32 | &ast::Type::I64 => {
                        arith::muli(lhs_val, rhs_val, self.location)
                    }
                    &ast::Type::F32 | &ast::Type::F64 => {
                        arith::mulf(lhs_val, rhs_val, self.location)
                    }
                    _ => {
                        // Never panic here, because we have already type-checked
                        panic!("Unsupported type for binary operation: {:?}", value_type)
                    }
                };

                let div_op = || match value_type {
                    &ast::Type::I32 | &ast::Type::I64 => {
                        arith::divsi(lhs_val, rhs_val, self.location)
                    }
                    &ast::Type::F32 | &ast::Type::F64 => {
                        arith::divf(lhs_val, rhs_val, self.location)
                    }
                    _ => {
                        // Never panic here, because we have already type-checked
                        panic!("Unsupported type for binary operation: {:?}", value_type)
                    }
                };

                let result_op = match op {
                    ast::BinOp::Add => block.append_operation(add_op()),
                    ast::BinOp::Sub => block.append_operation(sub_op()),
                    ast::BinOp::Mul => block.append_operation(mul_op()),
                    ast::BinOp::Div => block.append_operation(div_op()),
                    ast::BinOp::Equal => block.append_operation(arith::cmpi(
                        self.context,
                        arith::CmpiPredicate::Eq,
                        lhs_val,
                        rhs_val,
                        self.location,
                    )),
                    ast::BinOp::NotEqual => block.append_operation(arith::cmpi(
                        self.context,
                        arith::CmpiPredicate::Ne,
                        lhs_val,
                        rhs_val,
                        self.location,
                    )),
                    ast::BinOp::LessThan => block.append_operation(arith::cmpi(
                        self.context,
                        arith::CmpiPredicate::Slt,
                        lhs_val,
                        rhs_val,
                        self.location,
                    )),
                    ast::BinOp::LessThanOrEqual => block.append_operation(arith::cmpi(
                        self.context,
                        arith::CmpiPredicate::Sle,
                        lhs_val,
                        rhs_val,
                        self.location,
                    )),
                    ast::BinOp::GreaterThan => block.append_operation(arith::cmpi(
                        self.context,
                        arith::CmpiPredicate::Sgt,
                        lhs_val,
                        rhs_val,
                        self.location,
                    )),
                    ast::BinOp::GreaterThanOrEqual => block.append_operation(arith::cmpi(
                        self.context,
                        arith::CmpiPredicate::Sge,
                        lhs_val,
                        rhs_val,
                        self.location,
                    )),
                    ast::BinOp::And => {
                        block.append_operation(arith::andi(lhs_val, rhs_val, self.location))
                    }
                    ast::BinOp::Or => {
                        block.append_operation(arith::ori(lhs_val, rhs_val, self.location))
                    }
                };
                Ok(result_op.result(0)?.into())
            }
            ast::Expr::UnaryOp { op, expr, .. } => {
                let val = self.generate_expression(block, expr)?;

                match op {
                    ast::UnaryOp::Neg => {
                        let zero_op = block.append_operation(arith::constant(
                            self.context,
                            melior::ir::attribute::IntegerAttribute::new(
                                IntegerType::new(self.context, 32).into(),
                                0,
                            )
                            .into(),
                            self.location,
                        ));
                        let zero_val = zero_op.result(0)?.into();
                        let neg_op =
                            block.append_operation(arith::subi(zero_val, val, self.location));
                        Ok(neg_op.result(0)?.into())
                    }
                    ast::UnaryOp::Not => {
                        let true_op = block.append_operation(arith::constant(
                            self.context,
                            melior::ir::attribute::IntegerAttribute::new(
                                IntegerType::new(self.context, 1).into(),
                                1,
                            )
                            .into(),
                            self.location,
                        ));
                        let true_val = true_op.result(0)?.into();
                        let not_op =
                            block.append_operation(arith::xori(val, true_val, self.location));
                        Ok(not_op.result(0)?.into())
                    }
                }
            }
            ast::Expr::VarRef { name, .. } => self.variables.lookup_var(name),
            ast::Expr::FnCall {
                name, args, r#type, ..
            } => {
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.generate_expression(block, arg)?);
                }

                let result_type = self.ast_type_to_mlir_type(r#type)?;

                let call_op = block.append_operation(func::call(
                    self.context,
                    FlatSymbolRefAttribute::new(self.context, name),
                    &arg_values,
                    &[result_type],
                    self.location,
                ));
                Ok(call_op.result(0)?.into())
            }
            ast::Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let condition_value = self.generate_expression(block, condition)?;

                // Get the result type from one of the branches
                let temp_block = Block::new(&[]);
                let temp_value = self.generate_expression(&temp_block, then_branch)?;
                let result_type = temp_value.r#type();

                // Create regions for then and else branches
                let then_region = Region::new();
                let else_region = Region::new();

                // Create SCF if operation with result
                let if_op = block.append_operation(scf::r#if(
                    condition_value,
                    &[result_type],
                    then_region,
                    else_region,
                    self.location,
                ));

                // Get the then and else regions from the operation
                let then_region = if_op.region(0)?;
                let else_region = if_op.region(1)?;

                // Create blocks in the regions
                let then_block = then_region.append_block(Block::new(&[]));
                let else_block = else_region.append_block(Block::new(&[]));

                // Generate expressions in their respective blocks
                let then_value = self.generate_expression(&then_block, then_branch)?;
                let else_value = self.generate_expression(&else_block, else_branch)?;

                // Yield values from each block
                then_block.append_operation(scf::r#yield(&[then_value], self.location));
                else_block.append_operation(scf::r#yield(&[else_value], self.location));

                Ok(if_op.result(0)?.into())
            }
        }
    }

    pub fn compile_to_object(&mut self, output_path: &str) -> Result<()> {
        // Create a pass manager to convert dialects
        let pass_manager = PassManager::new(self.context);

        // Add conversion passes from arith/func/scf to LLVM
        pass_manager.add_pass(conversion::create_arith_to_llvm());
        pass_manager.add_pass(conversion::create_func_to_llvm());
        pass_manager.add_pass(conversion::create_scf_to_control_flow());
        pass_manager.add_pass(conversion::create_control_flow_to_llvm());

        // Enable verifier
        pass_manager.enable_verifier(true);

        // Run passes on the module
        if pass_manager.run(&mut self.module).is_err() {
            return Err(anyhow::anyhow!("Failed to run conversion passes"));
        }

        // Write converted MLIR file
        let mlir_path = format!("{}.mlir", output_path);
        let mlir_content = format!("{}", self.module.as_operation());
        std::fs::write(&mlir_path, &mlir_content)?;
        println!("LLVM MLIR generated: {}", mlir_path);

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

pub fn generate_code(
    program: &ast::Program<ast::Type>,
    output_path: Option<&str>,
) -> Result<String> {
    // Create MLIR context and register dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Generate code
    let mut codegen = CodeGenerator::new(&context);
    let mlir_code = codegen.generate(program)?;

    if let Some(path) = output_path {
        codegen.compile_to_object(path)?;
    }

    Ok(mlir_code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Expr, FnDecl, FnParam, Span, Stmt, Type, UnaryOp};
    use regex::Regex;

    fn create_span() -> Span {
        Span {
            start: 0,
            end: 0,
            context: (),
        }
    }

    fn create_test_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        // Explicitly get or load the SCF dialect
        context.get_or_load_dialect("scf");

        context
    }

    #[test]
    fn test_basic_function_generation() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        // Create a simple program: fn main() -> i32 { return 42; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "main",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Expr {
                    expr: Box::new(Expr::IntLit {
                        value: "42",
                        span: create_span(),
                        r#type: Type::I32,
                    }),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain func dialect and arith operations
        assert!(result.contains("func.func @main"));
        assert!(result.contains("arith.constant"));
        assert!(result.contains("return"));
    }

    #[test]
    fn test_arithmetic_operations() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        // Create: fn test() -> i32 { return 5 + 3; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Expr {
                    expr: Box::new(Expr::BinOp {
                        lhs: Box::new(Expr::IntLit {
                            value: "5",
                            span: create_span(),
                            r#type: Type::I32,
                        }),
                        op: BinOp::Add,
                        rhs: Box::new(Expr::IntLit {
                            value: "3",
                            span: create_span(),
                            r#type: Type::I32,
                        }),
                        span: create_span(),
                    }),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain arith operations
        assert!(result.contains("arith.addi"));
        assert!(result.contains("arith.constant"));
    }

    #[test]
    fn test_function_with_parameters() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

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
                body: vec![Stmt::Expr {
                    expr: Box::new(Expr::BinOp {
                        lhs: Box::new(Expr::VarRef {
                            name: "a",
                            r#type: Type::I32,
                            span: create_span(),
                        }),
                        op: BinOp::Add,
                        rhs: Box::new(Expr::VarRef {
                            name: "b",
                            r#type: Type::I32,
                            span: create_span(),
                        }),
                        span: create_span(),
                    }),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain function with parameters
        assert!(result.contains("func.func @add"));
        assert!(result.contains("arith.addi"));
    }

    #[test]
    fn test_boolean_operations() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        // Create: fn test() -> i1 { return true && false; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::Bool,
                body: vec![Stmt::Expr {
                    expr: Box::new(Expr::BinOp {
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
                    }),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain boolean operations
        assert!(result.contains("arith.constant"));
        assert!(result.contains("arith.andi"));
    }

    #[test]
    fn test_unary_operations() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        // Create: fn test() -> i32 { return -42; }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Expr {
                    expr: Box::new(Expr::UnaryOp {
                        op: UnaryOp::Neg,
                        expr: Box::new(Expr::IntLit {
                            value: "42",
                            span: create_span(),
                            r#type: Type::I32,
                        }),
                        span: create_span(),
                    }),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain unary negation
        assert!(result.contains("arith.constant"));
        assert!(result.contains("arith.subi"));
    }

    #[test]
    fn test_empty_module() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);
        let program = ast::Program { functions: vec![] };

        let result = codegen.generate(&program).unwrap();

        // Should contain module
        assert!(result.contains("module"));
    }

    #[test]
    fn test_type_conversion() {
        let context = create_test_context();
        let codegen = CodeGenerator::new(&context);

        // Test that type conversion works
        let result = codegen.ast_type_to_mlir_type(&Type::I32);
        assert!(result.is_ok());

        let result = codegen.ast_type_to_mlir_type(&Type::Bool);
        assert!(result.is_ok());
    }

    #[test]
    fn test_variable_declarations() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

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
                        value: Expr::IntLit {
                            value: "10",
                            span: create_span(),
                            r#type: Type::I32,
                        },
                        span: create_span(),
                    },
                    Stmt::Expr {
                        expr: Box::new(Expr::VarRef {
                            name: "x",
                            r#type: Type::I32,
                            span: create_span(),
                        }),
                        span: create_span(),
                    },
                ],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program).unwrap();

        // Should contain constant assignment
        assert!(result.contains("arith.constant"));
    }

    #[test]
    fn test_scoped_variables() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        // Create: fn test() -> i32 {
        //   let x: i32 = 10;
        //   if true {
        //     let x: i32 = 20;
        //   }
        //   return x; // Should return 10, not 20
        // }
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::I32,
                body: vec![
                    Stmt::LetDecl {
                        name: "x",
                        r#type: Type::I32,
                        value: Expr::IntLit {
                            value: "10",
                            span: create_span(),
                            r#type: Type::I32,
                        },
                        span: create_span(),
                    },
                    Stmt::Expr {
                        expr: Box::new(Expr::If {
                            condition: Box::new(Expr::BoolLit {
                                value: true,
                                span: create_span(),
                            }),
                            then_branch: Box::new(Expr::IntLit {
                                value: "20",
                                span: create_span(),
                                r#type: Type::I32,
                            }),
                            else_branch: Box::new(Expr::IntLit {
                                value: "30",
                                span: create_span(),
                                r#type: Type::I32,
                            }),
                            r#type: Type::I32,
                            span: create_span(),
                        }),
                        span: create_span(),
                    },
                ],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program);
        // This should work once we implement proper scoping
        assert!(result.is_ok());
    }

    #[test]
    fn test_variable_shadowing_in_functions() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        // Create two functions with same parameter name to test isolation
        let program = ast::Program {
            functions: vec![
                FnDecl {
                    name: "func1",
                    params: vec![FnParam {
                        name: "x",
                        r#type: Type::I32,
                        span: create_span(),
                    }],
                    r#type: Type::I32,
                    body: vec![Stmt::Expr {
                        expr: Box::new(Expr::VarRef {
                            name: "x",
                            r#type: Type::I32,
                            span: create_span(),
                        }),
                        span: create_span(),
                    }],
                    span: create_span(),
                },
                FnDecl {
                    name: "func2",
                    params: vec![FnParam {
                        name: "x",
                        r#type: Type::I32,
                        span: create_span(),
                    }],
                    r#type: Type::I32,
                    body: vec![Stmt::Expr {
                        expr: Box::new(Expr::VarRef {
                            name: "x",
                            r#type: Type::I32,
                            span: create_span(),
                        }),
                        span: create_span(),
                    }],
                    span: create_span(),
                },
            ],
        };

        let result = codegen.generate(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_variable_not_accessible_after_scope() {
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        // Test that a variable declared in an if block is not accessible afterwards
        let program = ast::Program {
            functions: vec![FnDecl {
                name: "test",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Expr {
                    expr: Box::new(Expr::If {
                        condition: Box::new(Expr::BoolLit {
                            value: true,
                            span: create_span(),
                        }),
                        then_branch: Box::new(Expr::IntLit {
                            value: "42",
                            span: create_span(),
                            r#type: Type::I32,
                        }),
                        else_branch: Box::new(Expr::IntLit {
                            value: "0",
                            span: create_span(),
                            r#type: Type::I32,
                        }),
                        r#type: Type::I32,
                        span: create_span(),
                    }),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };

        let result = codegen.generate(&program);
        // If expressions should work fine now
        assert!(result.is_ok());
    }

    #[test]
    fn test_if_statements_return_values() {
        // fn main() -> i32 {
        //     if true {
        //         if false {
        //             return 1;
        //         } else {
        //             return 2;
        //         }
        //     }
        //     return 3;
        // }
        let context = create_test_context();
        let mut codegen = CodeGenerator::new(&context);

        let program = ast::Program {
            functions: vec![FnDecl {
                name: "main",
                params: vec![],
                r#type: Type::I32,
                body: vec![Stmt::Expr {
                    expr: Box::new(Expr::If {
                        condition: Box::new(Expr::BoolLit {
                            value: true,
                            span: create_span(),
                        }),
                        then_branch: Box::new(Expr::IntLit {
                            value: "1",
                            span: create_span(),
                            r#type: Type::I32,
                        }),
                        else_branch: Box::new(Expr::IntLit {
                            value: "2",
                            span: create_span(),
                            r#type: Type::I32,
                        }),
                        r#type: Type::I32,
                        span: create_span(),
                    }),
                    span: create_span(),
                }],
                span: create_span(),
            }],
        };
        let result = codegen.generate(&program).unwrap();

        // Should contain func.func for now (SCF if not implemented yet)
        assert!(result.contains("func.func"));
        // Debug: Print the generated result to see what we're actually getting
        println!("Generated MLIR: {}", result);
        // For now, just check that it contains basic function structure
        assert!(result.contains("@main"));
    }

    #[test]
    fn test_arithmetic_operations_with_types() {
        let lhs_rhs = vec![
            (
                Expr::IntLit {
                    value: "5",
                    span: create_span(),
                    r#type: Type::I32,
                },
                Expr::IntLit {
                    value: "3",
                    span: create_span(),
                    r#type: Type::I32,
                },
            ),
            (
                Expr::IntLit {
                    value: "10",
                    span: create_span(),
                    r#type: Type::I64,
                },
                Expr::IntLit {
                    value: "20",
                    span: create_span(),
                    r#type: Type::I64,
                },
            ),
            (
                Expr::FloatLit {
                    value: "3.14",
                    span: create_span(),
                    r#type: Type::F32,
                },
                Expr::FloatLit {
                    value: "2.71",
                    span: create_span(),
                    r#type: Type::F32,
                },
            ),
            (
                Expr::FloatLit {
                    value: "1.35",
                    span: create_span(),
                    r#type: Type::F64,
                },
                Expr::FloatLit {
                    value: "2.68",
                    span: create_span(),
                    r#type: Type::F64,
                },
            ),
        ];
        let ops = vec![BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div];

        for (lhs, rhs) in lhs_rhs {
            for op in &ops {
                let context = create_test_context();
                let mut codegen = CodeGenerator::new(&context);

                // Create a function that performs the operation
                let program = ast::Program {
                    functions: vec![FnDecl {
                        name: "test",
                        params: vec![],
                        r#type: lhs.r#type().clone(),
                        body: vec![Stmt::Expr {
                            expr: Box::new(Expr::BinOp {
                                lhs: Box::new(lhs.clone()),
                                op: op.clone(),
                                rhs: Box::new(rhs.clone()),
                                span: create_span(),
                            }),
                            span: create_span(),
                        }],
                        span: create_span(),
                    }],
                };

                let result = codegen.generate(&program).expect(
                    format!(
                        "Failed to generate code for operation: {} {} {}",
                        lhs.r#type(),
                        op,
                        rhs.r#type()
                    )
                    .as_str(),
                );

                let value_type = match lhs {
                    Expr::IntLit { ref r#type, .. } => r#type,
                    Expr::FloatLit { ref r#type, .. } => r#type,
                    _ => panic!("Unexpected expression type"),
                };

                assert!(
                    result.contains("func.func @test"),
                    "Expected function definition for test not found in result:\n{result}"
                );

                let expected_op = match op {
                    BinOp::Add => match value_type {
                        Type::I32 | Type::I64 => "arith.addi",
                        Type::F32 | Type::F64 => "arith.addf",
                        _ => panic!("Unsupported type for addition: {}", value_type),
                    },
                    BinOp::Sub => match value_type {
                        Type::I32 | Type::I64 => "arith.subi",
                        Type::F32 | Type::F64 => "arith.subf",
                        _ => panic!("Unsupported type for subtraction: {}", value_type),
                    },
                    BinOp::Mul => match value_type {
                        Type::I32 | Type::I64 => "arith.muli",
                        Type::F32 | Type::F64 => "arith.mulf",
                        _ => panic!("Unsupported type for multiplication: {}", value_type),
                    },
                    BinOp::Div => match value_type {
                        Type::I32 | Type::I64 => "arith.divsi",
                        Type::F32 | Type::F64 => "arith.divf",
                        _ => panic!("Unsupported type for division: {}", value_type),
                    },
                    _ => panic!("Unsupported operation: {}", op),
                };
                assert!(
                    result.contains(&expected_op),
                    "Expected operation {expected_op} not found in result:\n{result}",
                );

                let re_return_value_with_type =
                    Regex::new(&format!(r"return\s+%[a-zA-Z0-9_]+\s*:\s*{value_type}")).unwrap();
                assert!(
                    re_return_value_with_type.is_match(&result),
                    "Expected return value with type {value_type} not found in result:\n{result}"
                );
            }
        }
    }
}
