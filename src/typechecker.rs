use std::collections::HashMap;

use crate::ast::{self, Span};

#[derive(Debug, Clone, serde::Serialize)]
pub enum TypeErrorKind<'a> {
    MismatchedTypes {
        left: ast::Type<'a>,
        right: ast::Type<'a>,
    },

    InvalidOperation {
        message: String,
    },

    DuplicateVariableDeclaration {
        name: &'a str,
    },

    DuplicateFunctionDeclaration {
        name: &'a str,
    },

    InvalidFunctionCall {
        message: String,
    },

    FunctionNotFound {
        name: &'a str,
    },

    VariableNotFound {
        name: &'a str,
    },

    AnnotationNotFound {
        message: String,
    },

    InternalError {
        message: String,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TypeError<'a> {
    pub kind: TypeErrorKind<'a>,
    #[allow(dead_code)]
    pub file_id: &'a str,
    pub span: Span,
}

impl std::fmt::Display for TypeError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            TypeErrorKind::MismatchedTypes { left, right } => {
                write!(f, "Mismatched types: {:?} and {:?}", left, right)
            }
            TypeErrorKind::InvalidOperation { message } => {
                write!(f, "Invalid operation: {}", message)
            }
            TypeErrorKind::DuplicateVariableDeclaration { name } => {
                write!(f, "Duplicate variable declaration: {}", name)
            }
            TypeErrorKind::DuplicateFunctionDeclaration { name } => {
                write!(f, "Duplicate function declaration: {}", name)
            }
            TypeErrorKind::InvalidFunctionCall { message } => {
                write!(f, "Invalid function call: {}", message)
            }
            TypeErrorKind::FunctionNotFound { name } => write!(f, "Function not found: {}", name),
            TypeErrorKind::VariableNotFound { name } => write!(f, "Variable not found: {}", name),
            TypeErrorKind::AnnotationNotFound { message } => {
                write!(f, "Annotation not found: {}", message)
            }
            TypeErrorKind::InternalError { message } => write!(f, "Internal error: {}", message),
        }
    }
}

fn assert_equal<'a>(
    left: ast::Type<'a>,
    right: ast::Type<'a>,
    file_id: &'a str,
    span: Span,
) -> Result<(), TypeError<'a>> {
    if left != right {
        Err(TypeError {
            kind: TypeErrorKind::MismatchedTypes { left, right },
            file_id,
            span,
        })
    } else {
        Ok(())
    }
}

fn duplicate_declaration_error<'a>(
    name: &'a str,
    r#type: ast::Type<'a>,
    file_id: &'a str,
    span: Span,
) -> TypeError<'a> {
    match r#type {
        ast::Type::Fn { .. } => TypeError {
            kind: TypeErrorKind::DuplicateFunctionDeclaration { name },
            file_id,
            span,
        },
        _ => TypeError {
            kind: TypeErrorKind::DuplicateVariableDeclaration { name },
            file_id,
            span,
        },
    }
}

fn not_found_error<'a>(
    name: &'a str,
    r#type: ast::Type<'a>,
    file_id: &'a str,
    span: Span,
) -> TypeError<'a> {
    match r#type {
        ast::Type::Fn { .. } => TypeError {
            kind: TypeErrorKind::FunctionNotFound { name },
            file_id,
            span,
        },
        _ => TypeError {
            kind: TypeErrorKind::VariableNotFound { name },
            file_id,
            span,
        },
    }
}

pub struct Variable<'ctx> {
    #[allow(dead_code)]
    name: &'ctx str,
    r#type: ast::Type<'ctx>,
}

pub struct Env<'ctx> {
    file_id: &'ctx str,
    scopes: Vec<HashMap<&'ctx str, Variable<'ctx>>>,
}

impl<'ctx> Env<'ctx> {
    pub fn new(file_id: &'ctx str) -> Self {
        Env {
            file_id,
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn declare_var(
        &mut self,
        name: &'ctx str,
        r#type: ast::Type<'ctx>,
        span: Span,
    ) -> Result<(), TypeError<'ctx>> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.contains_key(name) {
                return Err(duplicate_declaration_error(
                    name,
                    r#type,
                    self.file_id,
                    span,
                ));
            }
            scope.insert(name, Variable { name, r#type });
            Ok(())
        } else {
            Err(TypeError {
                kind: TypeErrorKind::InternalError {
                    message: "No scope available to declare variable".to_string(),
                },
                file_id: self.file_id,
                span,
            })
        }
    }

    pub fn lookup_var(
        &self,
        name: &'ctx str,
        r#type: ast::Type<'ctx>,
        span: Span,
    ) -> Result<&Variable<'ctx>, TypeError<'ctx>> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .ok_or_else(|| not_found_error(name, r#type, self.file_id, span))
    }

    pub fn lookup_fn(
        &self,
        name: &'ctx str,
        span: Span,
    ) -> Result<&Variable<'ctx>, TypeError<'ctx>> {
        let fn_var = self
            .scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .ok_or_else(|| TypeError {
                kind: TypeErrorKind::FunctionNotFound { name },
                file_id: self.file_id,
                span: span.clone(),
            })?;
        match fn_var.r#type {
            ast::Type::Fn { .. } => Ok(fn_var),
            _ => Err(TypeError {
                kind: TypeErrorKind::InvalidFunctionCall {
                    message: format!("{} is not a function", name),
                },
                file_id: self.file_id,
                span,
            }),
        }
    }
}

pub struct TypeChecker<'a> {
    env: Env<'a>,
    file_id: &'a str,
}

impl<'a> TypeChecker<'a> {
    pub fn new(file_id: &'a str) -> Self {
        TypeChecker {
            env: Env::new(file_id),
            file_id,
        }
    }

    pub fn typecheck_expr<'b>(
        &mut self,
        expr: &'b ast::Expr<'a>,
    ) -> Result<(ast::Type<'a>, ast::Expr<'a, ast::Type<'a>>), TypeError<'a>> {
        match expr {
            ast::Expr::IntLit {
                span,
                r#type,
                value,
            } => Ok((
                ast::Type::I32,
                ast::Expr::IntLit {
                    value,
                    span: span.clone(),
                    r#type: r#type.clone().unwrap_or(ast::Type::I32),
                },
            )),
            ast::Expr::BoolLit { span, value } => Ok((
                ast::Type::Bool,
                ast::Expr::BoolLit {
                    value: *value,
                    span: span.clone(),
                },
            )),
            ast::Expr::BinOp { lhs, op, rhs, span } => {
                let (lhs_type, lhs) = self.typecheck_expr(lhs)?;
                let (rhs_type, rhs) = self.typecheck_expr(rhs)?;
                match op {
                    ast::BinOp::Add | ast::BinOp::Sub | ast::BinOp::Mul | ast::BinOp::Div => {
                        assert_equal(
                            lhs_type.clone(),
                            rhs_type.clone(),
                            self.file_id,
                            span.clone(),
                        )?;
                        match lhs_type {
                            ast::Type::I32 | ast::Type::I64 | ast::Type::F32 | ast::Type::F64 => {
                                Ok((
                                    lhs_type,
                                    ast::Expr::BinOp {
                                        lhs: Box::new(lhs),
                                        op: op.clone(),
                                        rhs: Box::new(rhs),
                                        span: span.clone(),
                                    },
                                ))
                            }
                            _ => Err(TypeError {
                                kind: TypeErrorKind::InvalidOperation {
                                    message: format!(
                                        "Cannot perform arithmetic on type: {:?}",
                                        lhs_type
                                    ),
                                },
                                file_id: self.file_id,
                                span: span.clone(),
                            }),
                        }
                    }
                    ast::BinOp::Equal | ast::BinOp::NotEqual => {
                        assert_equal(
                            lhs_type.clone(),
                            rhs_type.clone(),
                            self.file_id,
                            span.clone(),
                        )?;
                        match lhs_type {
                            ast::Type::I32
                            | ast::Type::I64
                            | ast::Type::F32
                            | ast::Type::F64
                            | ast::Type::Bool => Ok((
                                ast::Type::Bool,
                                ast::Expr::BinOp {
                                    lhs: Box::new(lhs),
                                    op: op.clone(),
                                    rhs: Box::new(rhs),
                                    span: span.clone(),
                                },
                            )),
                            _ => Err(TypeError {
                                kind: TypeErrorKind::InvalidOperation {
                                    message: format!("Cannot compare type: {:?}", lhs_type),
                                },
                                file_id: self.file_id,
                                span: span.clone(),
                            }),
                        }
                    }
                    ast::BinOp::GreaterThan
                    | ast::BinOp::GreaterThanOrEqual
                    | ast::BinOp::LessThan
                    | ast::BinOp::LessThanOrEqual => {
                        assert_equal(
                            lhs_type.clone(),
                            rhs_type.clone(),
                            self.file_id,
                            span.clone(),
                        )?;
                        match lhs_type {
                            ast::Type::I32 | ast::Type::I64 | ast::Type::F32 | ast::Type::F64 => {
                                Ok((
                                    ast::Type::Bool,
                                    ast::Expr::BinOp {
                                        lhs: Box::new(lhs),
                                        op: op.clone(),
                                        rhs: Box::new(rhs),
                                        span: span.clone(),
                                    },
                                ))
                            }
                            _ => Err(TypeError {
                                kind: TypeErrorKind::InvalidOperation {
                                    message: format!("Cannot compare type: {:?}", lhs_type),
                                },
                                file_id: self.file_id,
                                span: span.clone(),
                            }),
                        }
                    }
                    ast::BinOp::And | ast::BinOp::Or => {
                        assert_equal(
                            lhs_type.clone(),
                            rhs_type.clone(),
                            self.file_id,
                            span.clone(),
                        )?;
                        match lhs_type {
                            ast::Type::Bool => Ok((
                                ast::Type::Bool,
                                ast::Expr::BinOp {
                                    lhs: Box::new(lhs),
                                    op: op.clone(),
                                    rhs: Box::new(rhs),
                                    span: span.clone(),
                                },
                            )),
                            _ => Err(TypeError {
                                kind: TypeErrorKind::InvalidOperation {
                                    message: format!(
                                        "Cannot perform logical operation on type: {:?}",
                                        lhs_type
                                    ),
                                },
                                file_id: self.file_id,
                                span: span.clone(),
                            }),
                        }
                    }
                }
            }
            ast::Expr::UnaryOp { op, expr, span } => {
                let (expr_type, expr) = self.typecheck_expr(expr)?;
                match op {
                    ast::UnaryOp::Neg => match expr_type {
                        ast::Type::I32 | ast::Type::I64 | ast::Type::F32 | ast::Type::F64 => Ok((
                            expr_type,
                            ast::Expr::UnaryOp {
                                op: op.clone(),
                                expr: Box::new(expr),
                                span: span.clone(),
                            },
                        )),
                        _ => Err(TypeError {
                            kind: TypeErrorKind::InvalidOperation {
                                message: format!("Cannot negate type: {:?}", expr_type),
                            },
                            file_id: self.file_id,
                            span: span.clone(),
                        }),
                    },
                    ast::UnaryOp::Not => {
                        if expr_type == ast::Type::Bool {
                            Ok((
                                ast::Type::Bool,
                                ast::Expr::UnaryOp {
                                    op: op.clone(),
                                    expr: Box::new(expr),
                                    span: span.clone(),
                                },
                            ))
                        } else {
                            Err(TypeError {
                                kind: TypeErrorKind::InvalidOperation {
                                    message: format!("Cannot negate type: {:?}", expr_type),
                                },
                                file_id: self.file_id,
                                span: span.clone(),
                            })
                        }
                    }
                }
            }
            ast::Expr::FnCall { name, args, span } => {
                let (return_type, params) = {
                    let fn_to_call = self.env.lookup_fn(name, span.clone())?;
                    match &fn_to_call.r#type {
                        ast::Type::Fn {
                            return_type,
                            params,
                        } => (*return_type.clone(), params.clone()),
                        _ => {
                            return Err(TypeError {
                                kind: TypeErrorKind::InvalidFunctionCall {
                                    message: format!("{} is not a function", name),
                                },
                                file_id: self.file_id,
                                span: span.clone(),
                            });
                        }
                    }
                };

                let mut typed_args = Vec::new();
                for arg in args {
                    let (arg_type, typed_arg) = self.typecheck_expr(arg)?;
                    typed_args.push((arg_type, typed_arg));
                }

                for ((arg_type, _), param) in typed_args.iter().zip(params.iter()) {
                    assert_equal(
                        arg_type.clone(),
                        param.r#type.clone(),
                        self.file_id,
                        param.span.clone(),
                    )?;
                }

                Ok((
                    return_type,
                    ast::Expr::FnCall {
                        name: *name,
                        args: typed_args.into_iter().map(|(_, arg)| arg).collect(),
                        span: span.clone(),
                    },
                ))
            }
            ast::Expr::VarRef { name, span } => {
                let var = self.env.lookup_var(name, ast::Type::Void, span.clone())?;
                Ok((
                    var.r#type.clone(),
                    ast::Expr::VarRef {
                        name: *name,
                        span: span.clone(),
                    },
                ))
            }
        }
    }

    pub fn typecheck_stmt<'b>(
        &mut self,
        stmt: &'b ast::Stmt<'a>,
    ) -> Result<(Option<ast::Type<'a>>, ast::Stmt<'a, ast::Type<'a>>), TypeError<'a>> {
        match stmt {
            ast::Stmt::FnDecl {
                name,
                params,
                r#type,
                body,
                span,
            } => {
                if let Some(return_type) = r#type {
                    let params = params
                        .iter()
                        .map(|param| {
                            let param_type = param.r#type.clone().ok_or_else(|| TypeError {
                                kind: TypeErrorKind::AnnotationNotFound {
                                    message: "Parameter type annotation is required".to_string(),
                                },
                                file_id: self.file_id,
                                span: param.span.clone(),
                            })?;
                            Ok(ast::FnParam {
                                name: param.name,
                                r#type: param_type,
                                span: param.span.clone(),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let fn_type = ast::Type::Fn {
                        return_type: Box::new(return_type.clone()),
                        params: params.clone(),
                    };
                    if self.env.lookup_fn(name, span.clone()).is_ok() {
                        return Err(duplicate_declaration_error(
                            name,
                            fn_type.clone(),
                            self.file_id,
                            span.clone(),
                        ));
                    }
                    self.env.declare_var(name, fn_type, span.clone())?;

                    // Push a new scope for the function body
                    self.env.push_scope();

                    // Declare parameters in the new scope
                    for param in &params {
                        self.env.declare_var(
                            param.name,
                            param.r#type.clone(),
                            param.span.clone(),
                        )?;
                    }

                    // Typecheck the function body
                    let mut typed_body = Vec::new();
                    for stmt in body {
                        typed_body.push(self.typecheck_stmt(stmt)?);
                    }

                    let actual_return_type = typed_body
                        .iter()
                        .last()
                        .and_then(|(ret_type, _)| ret_type.clone())
                        .unwrap_or(ast::Type::Void);

                    // Assert that the actual return type matches the declared return type
                    assert_equal(
                        return_type.clone(),
                        actual_return_type,
                        self.file_id,
                        span.clone(),
                    )?;

                    // Pop the scope for the function body
                    self.env.pop_scope();

                    Ok((
                        Some(return_type.clone()),
                        ast::Stmt::FnDecl {
                            name: *name,
                            params,
                            r#type: return_type.clone(),
                            body: typed_body.into_iter().map(|(_, stmt)| stmt).collect(),
                            span: span.clone(),
                        },
                    ))
                } else {
                    Err(TypeError {
                        kind: TypeErrorKind::AnnotationNotFound {
                            message: "Function return type annotation is required".to_string(),
                        },
                        file_id: self.file_id,
                        span: span.clone(),
                    })
                }
            }
            ast::Stmt::LetDecl {
                name,
                r#type,
                value,
                span,
            } => {
                let (actual_value_type, typed_value) = self.typecheck_expr(value)?;
                if let Some(r#type) = r#type {
                    assert_equal(
                        actual_value_type.clone(),
                        r#type.clone(),
                        self.file_id,
                        span.clone(),
                    )?;
                }
                let var_type = r#type.clone().unwrap_or(actual_value_type);
                self.env.declare_var(name, var_type.clone(), span.clone())?;
                Ok((
                    None,
                    ast::Stmt::LetDecl {
                        name: *name,
                        r#type: var_type,
                        value: typed_value,
                        span: span.clone(),
                    },
                ))
            }
            ast::Stmt::VarDecl {
                name,
                r#type,
                value,
                span,
            } => {
                let (actual_value_type, typed_value) = if let Some(value) = value {
                    let (actual_value_type, typed_value) = self.typecheck_expr(value)?;
                    (actual_value_type, Some(typed_value))
                } else {
                    (
                        r#type.clone().ok_or_else(|| TypeError {
                            kind: TypeErrorKind::AnnotationNotFound {
                                message: "Variable type annotation is required".to_string(),
                            },
                            file_id: self.file_id,
                            span: span.clone(),
                        })?,
                        None,
                    )
                };

                if let Some(r#type) = r#type {
                    assert_equal(
                        actual_value_type.clone(),
                        r#type.clone(),
                        self.file_id,
                        span.clone(),
                    )?;
                }

                let var_type = r#type.clone().unwrap_or(actual_value_type);

                self.env.declare_var(name, var_type.clone(), span.clone())?;
                Ok((
                    None,
                    ast::Stmt::VarDecl {
                        name: *name,
                        r#type: var_type,
                        value: typed_value,
                        span: span.clone(),
                    },
                ))
            }
            ast::Stmt::Assign { name, value, span } => {
                let var_type = {
                    let var = self.env.lookup_var(name, ast::Type::Void, span.clone())?;
                    var.r#type.clone()
                };
                let (actual_value_type, typed_value) = self.typecheck_expr(value)?;
                assert_equal(var_type, actual_value_type, self.file_id, span.clone())?;
                Ok((
                    None,
                    ast::Stmt::Assign {
                        name: *name,
                        value: Box::new(typed_value),
                        span: span.clone(),
                    },
                ))
            }
            ast::Stmt::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                let (actual_condition_type, typed_condition) = self.typecheck_expr(condition)?;
                assert_equal(
                    actual_condition_type,
                    ast::Type::Bool,
                    self.file_id,
                    span.clone(),
                )?;

                self.env.push_scope();
                let mut typed_then_branch = Vec::new();
                for stmt in then_branch {
                    let (_, typed_stmt) = self.typecheck_stmt(stmt)?;
                    typed_then_branch.push(typed_stmt);
                }
                self.env.pop_scope();

                let typed_else_branch = if let Some(else_branch) = else_branch {
                    // Push a new scope for the else branch
                    self.env.push_scope();
                    let mut typed_else_branch = Vec::new();
                    for stmt in else_branch {
                        let (_, typed_stmt) = self.typecheck_stmt(stmt)?;
                        typed_else_branch.push(typed_stmt);
                    }
                    self.env.pop_scope();
                    Some(typed_else_branch)
                } else {
                    None
                };

                Ok((
                    None,
                    ast::Stmt::If {
                        condition: Box::new(typed_condition),
                        then_branch: typed_then_branch,
                        else_branch: typed_else_branch,
                        span: span.clone(),
                    },
                ))
            }
            ast::Stmt::Return { expr, span } => {
                if let Some(expr) = expr {
                    let (actual_expr_type, typed_expr) = self.typecheck_expr(expr)?;
                    Ok((
                        Some(actual_expr_type),
                        ast::Stmt::Return {
                            expr: Some(Box::new(typed_expr)),
                            span: span.clone(),
                        },
                    ))
                } else {
                    Ok((
                        Some(ast::Type::Void),
                        ast::Stmt::Return {
                            expr: None,
                            span: span.clone(),
                        },
                    ))
                }
            }
            ast::Stmt::ExprStmt { expr, span } => {
                let (_, typed_expr) = self.typecheck_expr(expr)?;
                Ok((
                    None,
                    ast::Stmt::ExprStmt {
                        expr: Box::new(typed_expr),
                        span: span.clone(),
                    },
                ))
            }
            ast::Stmt::Expr { expr, .. } => {
                let (actual_expr_type, typed_expr) = self.typecheck_expr(expr)?;
                Ok((
                    Some(actual_expr_type),
                    ast::Stmt::Expr {
                        expr: Box::new(typed_expr),
                        span: expr.span().clone(),
                    },
                ))
            }
        }
    }

    pub fn typecheck_program<'b>(
        &mut self,
        program: &'b ast::Program<'a>,
    ) -> Result<ast::Program<'a, ast::Type<'a>>, TypeError<'a>> {
        let mut typed_functions = Vec::new();
        for function in &program.functions {
            // Convert FnDecl to Stmt::FnDecl for processing
            // TODO: Improve this by directly typechecking FnDecl
            let stmt = ast::Stmt::FnDecl {
                name: function.name,
                params: function.params.clone(),
                r#type: function.r#type.clone(),
                body: function.body.clone(),
                span: function.span.clone(),
            };
            let (_, typed_stmt) = self.typecheck_stmt(&stmt)?;
            match typed_stmt {
                ast::Stmt::FnDecl {
                    name,
                    params,
                    r#type,
                    body,
                    span,
                } => {
                    typed_functions.push(ast::FnDecl {
                        name,
                        params,
                        r#type,
                        body,
                        span,
                    });
                }
                _ => {
                    // Never expected this to happen, as we only process function declarations
                    return Err(TypeError {
                        kind: TypeErrorKind::InternalError {
                            message: "Expected a function declaration".to_string(),
                        },
                        file_id: self.file_id,
                        span: function.span.clone(),
                    });
                }
            }
        }
        Ok(ast::Program {
            functions: typed_functions,
        })
    }
}

#[allow(dead_code)]
pub fn typecheck<'a>(
    program: &ast::Program<'a>,
    file_id: &'a str,
) -> Result<ast::Program<'a, ast::Type<'a>>, TypeError<'a>> {
    let mut type_checker = TypeChecker::new(file_id);
    type_checker.typecheck_program(program)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use insta::assert_yaml_snapshot;

    use crate::{
        ast, parser,
        typechecker::{self, TypeError},
    };

    fn has_no_errors<T, E>(result: &chumsky::prelude::ParseResult<T, E>) -> bool {
        result.errors().len() == 0
    }

    fn typecheck(source: &str) -> Result<ast::Program<ast::Type>, TypeError> {
        let parse_result = parser::parse(source);
        assert!(has_no_errors(&parse_result));

        let program = parse_result.into_result().unwrap();

        typechecker::typecheck(&program, "<test>")
    }

    #[test]
    fn test_function_found() {
        let source = indoc! {"
            fn zero() -> i32 {
                return 0;
            }

            fn main() -> i32 {
                zero()
            }
        "};
        let result = typecheck(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_function_not_found() {
        let source = indoc! {"
            fn zero() -> i32 {
                return 0;
            }

            fn main() -> i32 {
                one()
            }
        "};
        let result = typecheck(source);
        assert!(result.is_err());
        assert_yaml_snapshot!(result);
    }

    #[test]
    fn test_variable_found() {
        let source = indoc! {"
            fn main() -> i32 {
                let x: i32 = 42;
                x
            }
        "};
        let result = typecheck(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_variable_not_found() {
        let source = indoc! {"
            fn main() -> i32 {
                let x: i32 = 42;
                y
            }
        "};
        let result = typecheck(source);
        assert!(result.is_err());
        assert_yaml_snapshot!(result);
    }

    #[test]
    fn test_function_declaration() {
        let source = indoc! {"
            fn add() -> i32 {
                let x: i32 = 42;
                let y: i32 = 43;
                x + y
            }

            fn main() -> i32 {
                let x: i32 = 42;
                let y: i32 = 43;

                if true {
                    let x: i32 = 44;
                } else {
                    let y: i32 = 45;
                }
                
                x + y
            }
        "};
        let result = typecheck(source);
        assert!(result.is_ok(), "Expected ok but got error: {:?}", result);
    }

    #[test]
    fn test_duplicate_variable_declaration() {
        let source = indoc! {"
            fn main() -> i32 {
                let x: i32 = 42;
                let x: i32 = 43;
                x
            }
        "};
        let result = typecheck(source);
        assert!(result.is_err());
        assert_yaml_snapshot!(result);
    }

    #[test]
    fn test_duplicate_function_declaration() {
        let source = indoc! {"
            fn add() -> i32 {
                return 0;
            }

            fn add() -> i32 {
                return 1;
            }
        "};
        let result = typecheck(source);
        assert!(result.is_err());
        assert_yaml_snapshot!(result);
    }

    #[test]
    fn test_mismatched_types() {
        let source = indoc! {"
            fn main() -> i32 {
                let x: i32 = true;
                0
            }
        "};
        let result = typecheck(source);
        assert!(result.is_err());
        assert_yaml_snapshot!(result);
    }

    #[test]
    fn test_matched_types() {
        let source = indoc! {"
            fn main() -> i32 {
                let x: i32 = 42;
                let y: i32 = 43;
                x + y
            }
        "};
        let result = typecheck(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_operation() {
        let source = indoc! {"
            fn main() -> i32 {
                1 + true
            }
        "};
        let result = typecheck(source);
        assert!(result.is_err());
        assert_yaml_snapshot!(result);
    }

    #[test]
    fn test_valid_operation() {
        let source = indoc! {"
            fn main() -> i32 {
                1 + 2
            }
        "};
        let result = typecheck(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_function_call() {
        let source = indoc! {"
            fn add(x: i32, y: i32) -> i32 {
                x + y
            }

            fn main() -> i32 {
                add(1, true)
            }
        "};
        let result = typecheck(source);
        assert!(result.is_err());
        assert_yaml_snapshot!(result);
    }

    #[test]
    fn test_valid_function_call() {
        let source = indoc! {"
            fn add(x: i32, y: i32) -> i32 {
                x + y
            }

            fn main() -> i32 {
                add(1, 2)
            }
        "};
        let result = typecheck(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_annotation_found() {
        let source = indoc! {"
            fn add(x: i32, y: i32) -> i32 {
                x + y
            }
            
            fn main() -> i32 {
                0
            }
        "};
        let result = typecheck(source);
        assert!(
            result.is_ok(),
            "Type checking should succeed, but got error: {:?}",
            result
        );
    }
}
