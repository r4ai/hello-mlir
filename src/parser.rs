use chumsky::{input::ValueInput, prelude::*};
use logos::Logos;

use crate::{ast, token::Token};

pub fn parser<'a, I>() -> impl Parser<'a, I, ast::Program<'a>, extra::Err<Rich<'a, Token<'a>>>>
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    let identifier = select! {
        Token::Identifier(value) => value
    };

    let r#type = recursive(|r#type| {
        choice((
            select! {
                Token::Identifier(value) if value == "bool" => ast::Type::Bool,
                Token::Identifier(value) if value == "i32" => ast::Type::I32,
                Token::Identifier(value) if value == "i64" => ast::Type::I64,
                Token::Identifier(value) if value == "f32" => ast::Type::F32,
                Token::Identifier(value) if value == "f64" => ast::Type::F64,
                Token::Identifier(value) if value == "void" => ast::Type::Void,
                Token::Identifier(value) if value == "string" => ast::Type::String,
            },
            // "(" [ { identifier ":" type "," } identifier ":" type ] ")" "->" type
            identifier
                .then_ignore(just(Token::Colon))
                .then(r#type.clone())
                .map_with(|(name, ty), e| ast::FnParam {
                    name,
                    r#type: ty,
                    span: ast::Span::from(e.span()),
                })
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .then_ignore(just(Token::RightArrow))
                .then(r#type)
                .map(|(params, return_type)| ast::Type::Fn {
                    return_type: Box::new(return_type),
                    params,
                }),
        ))
    })
    .boxed();

    // The operator precedence and associativity are designed to match C++ according to:
    // https://www.ibm.com/docs/en/i/7.3.0?topic=operators-operator-precedence-associativity
    let expr = recursive(|expr| {
        let literal = select! {
            Token::Integer(value) = e => ast::Expr::IntLit {
                value: value.parse().unwrap(),
                span: ast::Span::from(e.span())
            },
            Token::Identifier(ident) = e if ident == "true" => ast::Expr::BoolLit {
                value: true,
                span: ast::Span::from(e.span())
            },
            Token::Identifier(ident) = e if ident == "false" => ast::Expr::BoolLit {
                value: false,
                span: ast::Span::from(e.span())
            },
        };

        // variable reference (identifier as expression)
        let var_ref = identifier.map_with(|name, e| ast::Expr::VarRef {
            name,
            span: ast::Span::from(e.span()),
        });

        // "(" [ { expr "," } expr ] ")"
        let call_args = expr
            .clone()
            .separated_by(just(Token::Comma))
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // function call: identifier '(' [args] ')' (only in expression context)
        let function_call =
            identifier
                .then(call_args)
                .map_with(|(name, args), e| ast::Expr::FnCall {
                    name,
                    args,
                    span: ast::Span::from(e.span()),
                });

        let primary = choice((
            // function call
            function_call,
            // literal
            literal,
            // variable reference
            var_ref,
            // "(" expr ")"
            expr.clone()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        ));

        let unary = choice((
            // "-" primary
            just(Token::Sub)
                .ignore_then(primary.clone())
                .map_with(|expr, e| ast::Expr::UnaryOp {
                    op: ast::UnaryOp::Neg,
                    expr: Box::new(expr),
                    span: ast::Span::from(e.span()),
                }),
            // "!" primary
            just(Token::Not)
                .ignore_then(primary.clone())
                .map_with(|expr, e| ast::Expr::UnaryOp {
                    op: ast::UnaryOp::Not,
                    expr: Box::new(expr),
                    span: ast::Span::from(e.span()),
                }),
            // primary
            primary,
        ));

        // unary { ("*" | "/") unary }
        let multiplication = unary.clone().foldl_with(
            choice((
                just(Token::Mul).to(ast::BinOp::Mul),
                just(Token::Div).to(ast::BinOp::Div),
            ))
            .then(unary)
            .repeated(),
            |lhs, (op, rhs), e| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
                span: ast::Span::from(e.span()),
            },
        );

        // multiplication { ("+" | "-") multiplication }
        let addition = multiplication
            .clone()
            .foldl_with(
                choice((
                    just(Token::Add).to(ast::BinOp::Add),
                    just(Token::Sub).to(ast::BinOp::Sub),
                ))
                .then(multiplication)
                .repeated(),
                |lhs, (op, rhs), e| ast::Expr::BinOp {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                    span: ast::Span::from(e.span()),
                },
            )
            .boxed();

        // addition { ("<" | "<=" | ">" | ">=") addition }
        let comparison = addition.clone().foldl_with(
            choice((
                just(Token::Equal).to(ast::BinOp::Equal),
                just(Token::NotEqual).to(ast::BinOp::NotEqual),
                just(Token::LessThan).to(ast::BinOp::LessThan),
                just(Token::LessThanOrEqual).to(ast::BinOp::LessThanOrEqual),
                just(Token::GreaterThan).to(ast::BinOp::GreaterThan),
                just(Token::GreaterThanOrEqual).to(ast::BinOp::GreaterThanOrEqual),
            ))
            .then(addition)
            .repeated(),
            |lhs, (op, rhs), e| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
                span: ast::Span::from(e.span()),
            },
        );

        // comparison { ("==" | "!=") comparison }
        let equality = comparison
            .clone()
            .foldl_with(
                choice((
                    just(Token::Equal).to(ast::BinOp::Equal),
                    just(Token::NotEqual).to(ast::BinOp::NotEqual),
                ))
                .then(comparison)
                .repeated(),
                |lhs, (op, rhs), e| ast::Expr::BinOp {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                    span: ast::Span::from(e.span()),
                },
            )
            .boxed();

        // equality { "&&" equality }
        let logical_and = equality.clone().foldl_with(
            just(Token::And)
                .to(ast::BinOp::And)
                .then(equality)
                .repeated(),
            |lhs, (op, rhs), e| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
                span: ast::Span::from(e.span()),
            },
        );

        // logical_and { "||" logical_and }
        #[allow(clippy::let_and_return)]
        let logical_or = logical_and.clone().foldl_with(
            just(Token::Or)
                .to(ast::BinOp::Or)
                .then(logical_and)
                .repeated(),
            |lhs, (op, rhs), e| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
                span: ast::Span::from(e.span()),
            },
        );

        logical_or
    })
    .boxed();

    let statements = recursive(|statements| {
        // expr ";"
        let expr_statement =
            expr.clone()
                .then_ignore(just(Token::Semicolon))
                .map_with(|expr, e| ast::Stmt::ExprStmt {
                    expr: Box::new(expr),
                    span: ast::Span::from(e.span()),
                });

        // "let" identifier [":" type] ["=" expr] ";"
        let let_declaration = just(Token::LetDeclaration)
            .ignore_then(identifier)
            .then(just(Token::Colon).ignore_then(r#type.clone()).or_not())
            .then(just(Token::Assign).ignore_then(expr.clone()).or_not())
            .then_ignore(just(Token::Semicolon))
            .map_with(|((name, ty), value), e| ast::Stmt::LetDecl {
                name,
                r#type: ty,
                value,
                span: ast::Span::from(e.span()),
            });

        // "var" identifier [":" type] ["=" expr] ";"
        let var_declaration = just(Token::VarDeclaration)
            .ignore_then(identifier)
            .then(just(Token::Colon).ignore_then(r#type.clone()).or_not())
            .then(just(Token::Assign).ignore_then(expr.clone()).or_not())
            .then_ignore(just(Token::Semicolon))
            .map_with(|((name, ty), value), e| ast::Stmt::VarDecl {
                name,
                r#type: ty,
                value,
                span: ast::Span::from(e.span()),
            });

        // identifier "=" expr ";"
        let assignment = identifier
            .then_ignore(just(Token::Assign))
            .then(expr.clone())
            .then_ignore(just(Token::Semicolon))
            .map_with(|(name, value), e| ast::Stmt::Assign {
                name,
                value: Box::new(value),
                span: ast::Span::from(e.span()),
            });

        // "return" [ expr ] ";"
        let return_statement = just(Token::Return)
            .ignore_then(expr.clone().or_not())
            .then_ignore(just(Token::Semicolon))
            .map_with(|expr, e| ast::Stmt::Return {
                expr: expr.map(Box::new),
                span: ast::Span::from(e.span()),
            });

        // identifier ":" type
        let function_parameter = identifier
            .then_ignore(just(Token::Colon))
            .then(r#type.clone())
            .map_with(|(name, ty), e| ast::FnParam {
                name,
                r#type: Some(ty),
                span: ast::Span::from(e.span()),
            });

        // "(" { function_parameter "," } function_parameter ")"
        let function_parameters = just(Token::LParen)
            .ignore_then(
                function_parameter
                    .separated_by(just(Token::Comma))
                    .collect::<Vec<_>>(),
            )
            .then_ignore(just(Token::RParen));

        // "{" statements [ expr ] "}"
        let block = just(Token::LBrace)
            .ignore_then(statements.clone())
            .then_ignore(just(Token::RBrace));

        // "fn" identifier function_parameters "->" type function_body
        let function_declaration = just(Token::FunctionDeclaration)
            .ignore_then(identifier)
            .then(function_parameters)
            .then_ignore(just(Token::RightArrow))
            .then(r#type.clone())
            .then(block.clone())
            .map_with(
                |(((name, params), return_type), body), e| ast::Stmt::FnDecl {
                    name,
                    params,
                    r#type: Some(return_type),
                    body,
                    span: ast::Span::from(e.span()),
                },
            );

        // "if" expr block [ "else" (if_stmt | block) ]
        let if_statement = recursive(|if_stmt| {
            just(Token::If)
                .ignore_then(expr.clone())
                .then(block.clone())
                .then(
                    just(Token::Else)
                        .ignore_then(
                            if_stmt
                                .clone()
                                .map_with(|stmt, e| match stmt {
                                    ast::Stmt::If {
                                        condition,
                                        then_branch,
                                        else_branch,
                                        ..
                                    } => vec![ast::Stmt::If {
                                        condition,
                                        then_branch,
                                        else_branch,
                                        span: ast::Span::from(e.span()),
                                    }],
                                    _ => unreachable!(), // This will always be an If statement
                                })
                                .or(block.clone()),
                        )
                        .or_not(),
                )
                .map_with(|((condition, then_branch), else_branch), e| ast::Stmt::If {
                    condition: Box::new(condition),
                    then_branch,
                    else_branch,
                    span: ast::Span::from(e.span()),
                })
        });

        let statement = choice((
            let_declaration,
            var_declaration,
            assignment,
            return_statement,
            function_declaration,
            expr_statement,
            if_statement,
        ));

        statement
            .repeated()
            .collect::<Vec<_>>()
            .then(expr.clone().or_not().map(|expr| {
                expr.map(|expr| {
                    let span = expr.span().clone();
                    ast::Stmt::Expr {
                        expr: Box::new(expr),
                        span,
                    }
                })
            }))
            .map(|(statements, expr)| {
                let mut body: Vec<ast::Stmt> = statements;
                if let Some(expr) = expr {
                    body.push(expr);
                }
                body
            })
    })
    .boxed();

    // identifier ":" type
    let function_parameter = identifier
        .then_ignore(just(Token::Colon))
        .then(r#type.clone())
        .map_with(|(name, ty), e| ast::FnParam {
            name,
            r#type: Some(ty),
            span: ast::Span::from(e.span()),
        });

    // "(" { function_parameter "," } function_parameter ")"
    let function_parameters = just(Token::LParen)
        .ignore_then(
            function_parameter
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RParen));

    // Top-level function declaration (only allowed at file level)
    let top_level_function_declaration = just(Token::FunctionDeclaration)
        .ignore_then(identifier)
        .then(function_parameters)
        .then_ignore(just(Token::RightArrow))
        .then(r#type.clone())
        .then(
            just(Token::LBrace)
                .ignore_then(statements.clone())
                .then_ignore(just(Token::RBrace)),
        )
        .map_with(|(((name, params), return_type), body), e| ast::FnDecl {
            name,
            params,
            r#type: Some(return_type),
            body,
            span: ast::Span::from(e.span()),
        });

    // Parse only function declarations at the top level
    top_level_function_declaration
        .repeated()
        .collect::<Vec<_>>()
        .then_ignore(end())
        .map(|functions: Vec<ast::FnDecl<'_, Option<ast::Type>>>| ast::Program { functions })
}

pub fn parse(src: &str) -> ParseResult<ast::Program, chumsky::error::Rich<'_, Token<'_>>> {
    // Create a logos lexer over the source code
    let token_iter = Token::lexer(src)
        .spanned()
        // Convert logos errors into tokens. We want parsing to be recoverable and not fail at the lexing stage, so
        // we have a dedicated `Token::Error` variant that represents a token error that was previously encountered
        .map(|(tok, span)| match tok {
            // Turn the `Range<usize>` spans logos gives us into chumsky's `SimpleSpan` via `Into`, because it's easier
            // to work with
            Ok(tok) => (tok, span.into()),
            Err(()) => (Token::Error, span.into()),
        });

    // Turn the token iterator into a stream that chumsky can use for things like backtracking
    let token_stream = chumsky::input::Stream::from_iter(token_iter)
        // Tell chumsky to split the (Token, SimpleSpan) stream into its parts so that it can handle the spans for us
        // This involves giving chumsky an 'end of input' span: we just use a zero-width span at the end of the string
        .map((0..src.len()).into(), |(t, s): (_, _)| (t, s));

    // Parse the token stream with our chumsky parser
    parser().parse(token_stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use insta::assert_yaml_snapshot;

    // Helper function to check if a parse result has no errors
    fn has_no_errors<T, E>(result: &ParseResult<T, E>) -> bool {
        result.errors().len() == 0
    }

    #[test]
    fn test_parse_integer_literal() {
        let input = "fn test() -> i32 { 42 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_binary_expression() {
        let input = "fn test() -> i32 { 42 + 10 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_complex_expression() {
        let input = "fn test() -> i32 { 42 + (10 * 5) - 8 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_unary_expression() {
        let input = "fn test() -> i32 { -42 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_function_declaration() {
        let input = "fn zero() -> i32 { 0 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_function_with_multiple_statements() {
        let input = "fn compute() -> i32 { 10 + 20; 30 + 40 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_function_call() {
        let input = indoc! {"
            fn zero() -> i32 { 0 }
            fn main() -> i32 { zero() }  
        "};
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_variable_declaration() {
        let input = "fn test() -> i32 { let x: i32 = 42; x }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_variable_declaration_without_type() {
        let input = "fn test() -> i32 { let x = 42; x }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_variable_declaration_without_value() {
        let input = "fn test() -> i32 { let x: i32; 42 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_mutable_variable_declaration() {
        let input = "fn test() -> i32 { var x: i32 = 42; x }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_mutable_variable_declaration_without_type() {
        let input = "fn test() -> i32 { var x = 42; x }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_mutable_variable_declaration_without_value() {
        let input = "fn test() -> i32 { var x: i32; 42 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_assignment() {
        let input = "fn test() -> i32 { var x: i32 = 0; x = 42; x }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_if_statement() {
        let input = "fn test() -> i32 { if 1 { 1 } else { 0 } }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_if_statement_with_else() {
        let input = "fn test() -> i32 { if 1 { 1 } else { 2 } }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_if_statement_with_nested_else() {
        let input = indoc! {"
            fn test() -> i32 {
                if 1 {
                    if 2 {
                        3
                    } else {
                        4
                    }
                } else {
                    5
                }
            }
        "};
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_if_statement_with_else_if_chain() {
        let input = indoc! {"
            fn test() -> i32 {
                if 1 {
                    1
                } else if 2 {
                    2
                } else if 3 {
                    3
                } else {
                    4
                }
            }
        "};
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_error_recovery() {
        let input = "42 + (10 * 5 - 8";
        let result = parse(input);

        // This should contain errors
        assert!(!has_no_errors(&result));
        let errors = result.into_errors();
        assert_yaml_snapshot!(format!("{:?}", errors));
    }

    #[test]
    fn test_parse_with_comments() {
        let input = indoc! {"
            // Line comment
            fn run() -> i32 {
                // Another line comment
                let a: i32 = /* block comment */ 10;
                let b: i32 = 20; // Comment at end of line

                /* Multi-line
                   block comment
                   with multiple lines */
                a + b // Result with comment
            }
        "};
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    // Tests for new grammar constraint: only function declarations at file level
    #[test]
    fn test_parse_top_level_let_declaration_should_fail() {
        let input = "let x: i32 = 42;";
        let result = parse(input);

        // This should now fail with our new grammar
        assert!(!has_no_errors(&result));
        let errors = result.into_errors();
        assert_yaml_snapshot!(format!("{:?}", errors));
    }

    #[test]
    fn test_parse_top_level_expression_should_fail() {
        let input = "42 + 10";
        let result = parse(input);

        // This should now fail with our new grammar
        assert!(!has_no_errors(&result));
        let errors = result.into_errors();
        assert_yaml_snapshot!(format!("{:?}", errors));
    }

    #[test]
    fn test_parse_top_level_assignment_should_fail() {
        let input = "x = 42;";
        let result = parse(input);

        // This should now fail with our new grammar
        assert!(!has_no_errors(&result));
        let errors = result.into_errors();
        assert_yaml_snapshot!(format!("{:?}", errors));
    }

    #[test]
    fn test_parse_only_function_declarations_should_pass() {
        let input = indoc! {"
            fn zero() -> i32 { 0 }
            fn one() -> i32 { 1 }
            fn add(x: i32, y: i32) -> i32 { x + y }
        "};
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_mixed_top_level_statements_should_fail() {
        let input = indoc! {"
            fn zero() -> i32 { 0 }
            let x: i32 = 42;
            fn one() -> i32 { 1 }
        "};
        let result = parse(input);

        // This should now fail with our new grammar
        assert!(!has_no_errors(&result));
        let errors = result.into_errors();
        assert_yaml_snapshot!(format!("{:?}", errors));
    }

    #[test]
    fn test_parse_empty_file_should_pass() {
        let input = "";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert!(program.functions.is_empty());
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_functions_with_comments_should_pass() {
        let input = indoc! {"
            // First function
            fn zero() -> i32 { 0 }
            
            /* Block comment */
            fn one() -> i32 { 
                // Comment inside function
                1 
            }
        "};
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_eq!(program.functions.len(), 2);
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_top_level_if_statement_should_fail() {
        let input = "if true { 42 } else { 0 }";
        let result = parse(input);

        // This should now fail with our new grammar
        assert!(!has_no_errors(&result));
        let errors = result.into_errors();
        assert_yaml_snapshot!(format!("{:?}", errors));
    }
}
