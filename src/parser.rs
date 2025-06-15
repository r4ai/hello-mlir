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

    let r#type = select! {
        Token::Identifier(value) if value == "i32" => ast::Type::I32,
        Token::Identifier(value) if value == "i64" => ast::Type::I64,
        Token::Identifier(value) if value == "f32" => ast::Type::F32,
        Token::Identifier(value) if value == "f64" => ast::Type::F64,
        Token::Identifier(value) if value == "void" => ast::Type::Void,
        Token::Identifier(value) if value == "string" => ast::Type::String,
    };

    // The operator precedence and associativity are designed to match C++ according to:
    // https://www.ibm.com/docs/en/i/7.3.0?topic=operators-operator-precedence-associativity
    let expr = recursive(|expr| {
        let literal = select! {
            Token::Integer(value) => ast::Expr::IntLit(value.parse().unwrap()),
            Token::Identifier(ident) if ident == "true" => ast::Expr::BoolLit(true),
            Token::Identifier(ident) if ident == "false" => ast::Expr::BoolLit(false),
        };

        // variable reference (identifier as expression)
        let var_ref = identifier.map(|name| ast::Expr::VarRef { name });

        // "(" [ { expr "," } expr ] ")"
        let call_args = expr
            .clone()
            .separated_by(just(Token::Comma))
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // function call: identifier '(' [args] ')' (only in expression context)
        let function_call = identifier
            .then(call_args)
            .map(|(name, args)| ast::Expr::FnCall { name, args });

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
                .map(|expr| ast::Expr::UnaryOp {
                    op: ast::UnaryOp::Neg,
                    expr: Box::new(expr),
                }),
            // "!" primary
            just(Token::Not)
                .ignore_then(primary.clone())
                .map(|expr| ast::Expr::UnaryOp {
                    op: ast::UnaryOp::Not,
                    expr: Box::new(expr),
                }),
            // primary
            primary,
        ));

        // unary { ("*" | "/") unary }
        let multiplication = unary.clone().foldl(
            choice((
                just(Token::Mul).to(ast::BinOp::Mul),
                just(Token::Div).to(ast::BinOp::Div),
            ))
            .then(unary)
            .repeated(),
            |lhs, (op, rhs)| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            },
        );

        // multiplication { ("+" | "-") multiplication }
        let addition = multiplication
            .clone()
            .foldl(
                choice((
                    just(Token::Add).to(ast::BinOp::Add),
                    just(Token::Sub).to(ast::BinOp::Sub),
                ))
                .then(multiplication)
                .repeated(),
                |lhs, (op, rhs)| ast::Expr::BinOp {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
            )
            .boxed();

        // addition { ("<" | "<=" | ">" | ">=") addition }
        let comparison = addition.clone().foldl(
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
            |lhs, (op, rhs)| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            },
        );

        // comparison { ("==" | "!=") comparison }
        let equality = comparison
            .clone()
            .foldl(
                choice((
                    just(Token::Equal).to(ast::BinOp::Equal),
                    just(Token::NotEqual).to(ast::BinOp::NotEqual),
                ))
                .then(comparison)
                .repeated(),
                |lhs, (op, rhs)| ast::Expr::BinOp {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
            )
            .boxed();

        // equality { "&&" equality }
        let logical_and = equality.clone().foldl(
            just(Token::And)
                .to(ast::BinOp::And)
                .then(equality)
                .repeated(),
            |lhs, (op, rhs)| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            },
        );

        // logical_and { "||" logical_and }
        #[allow(clippy::let_and_return)]
        let logical_or = logical_and.clone().foldl(
            just(Token::Or)
                .to(ast::BinOp::Or)
                .then(logical_and)
                .repeated(),
            |lhs, (op, rhs)| ast::Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            },
        );

        logical_or
    });

    let statements = recursive(|statements| {
        // expr ";"
        let expr_statement = expr
            .clone()
            .then_ignore(just(Token::Semicolon))
            .map(|expr| ast::Stmt::ExprStmt {
                expr: Box::new(expr),
            });

        // "let" identifier [":" type] ["=" expr] ";"
        let let_declaration = just(Token::LetDeclaration)
            .ignore_then(identifier)
            .then(just(Token::Colon).ignore_then(r#type).or_not())
            .then(just(Token::Assign).ignore_then(expr.clone()).or_not())
            .then_ignore(just(Token::Semicolon))
            .map(|((name, ty), value)| ast::Stmt::LetDecl {
                name,
                r#type: ty,
                value,
            });

        // "var" identifier [":" type] ["=" expr] ";"
        let var_declaration = just(Token::VarDeclaration)
            .ignore_then(identifier)
            .then(just(Token::Colon).ignore_then(r#type).or_not())
            .then(just(Token::Assign).ignore_then(expr.clone()).or_not())
            .then_ignore(just(Token::Semicolon))
            .map(|((name, ty), value)| ast::Stmt::VarDecl {
                name,
                r#type: ty,
                value,
            });

        // identifier "=" expr ";"
        let assignment = identifier
            .then_ignore(just(Token::Assign))
            .then(expr.clone())
            .then_ignore(just(Token::Semicolon))
            .map(|(name, value)| ast::Stmt::Assign {
                name,
                value: Box::new(value),
            });

        // "return" [ expr ] ";"
        let return_statement = just(Token::Return)
            .ignore_then(expr.clone().or_not())
            .then_ignore(just(Token::Semicolon))
            .map(|expr| ast::Stmt::Return {
                expr: expr.map(Box::new),
            });

        // identifier ":" type
        let function_parameter =
            identifier
                .then_ignore(just(Token::Colon))
                .then(r#type)
                .map(|(name, ty)| ast::FnParam {
                    name,
                    r#type: Some(ty),
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
            .then(r#type)
            .then(block.clone())
            .map(|(((name, params), return_type), body)| ast::Stmt::FnDecl {
                name,
                params,
                r#type: Some(return_type),
                body,
            });

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
                                .map(|stmt| match stmt {
                                    ast::Stmt::If {
                                        condition,
                                        then_branch,
                                        else_branch,
                                    } => vec![ast::Stmt::If {
                                        condition,
                                        then_branch,
                                        else_branch,
                                    }],
                                    _ => unreachable!(), // This will always be an If statement
                                })
                                .or(block.clone()),
                        )
                        .or_not(),
                )
                .map(|((condition, then_branch), else_branch)| ast::Stmt::If {
                    condition: Box::new(condition),
                    then_branch,
                    else_branch,
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
                expr.map(|expr| ast::Stmt::Expr {
                    expr: Box::new(expr),
                })
            }))
            .map(|(statements, expr)| {
                let mut body: Vec<ast::Stmt> = statements;
                if let Some(expr) = expr {
                    body.push(expr);
                }
                body
            })
    });

    statements
        .then_ignore(end())
        .map(|stmts: Vec<ast::Stmt<'_, Option<ast::Type>>>| ast::Program { stmts })
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
        let input = "42";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_binary_expression() {
        let input = "42 + 10";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_complex_expression() {
        let input = "42 + (10 * 5) - 8";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_unary_expression() {
        let input = "-42";
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
            zero()
        "};
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_variable_declaration() {
        let input = "let x: i32 = 42;";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_variable_declaration_without_type() {
        let input = "let x = 42;";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_variable_declaration_without_value() {
        let input = "let x: i32;";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_mutable_variable_declaration() {
        let input = "var x: i32 = 42;";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_mutable_variable_declaration_without_type() {
        let input = "var x = 42;";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_mutable_variable_declaration_without_value() {
        let input = "var x: i32;";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_assignment() {
        let input = "x = 42;";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_if_statement() {
        let input = "if 1 { 1 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_if_statement_with_else() {
        let input = "if 1 { 1 } else { 2 }";
        let result = parse(input);
        assert!(has_no_errors(&result));

        let program = result.into_result().unwrap();
        assert_yaml_snapshot!(program);
    }

    #[test]
    fn test_parse_if_statement_with_nested_else() {
        let input = indoc! {"
            if 1 {
                if 2 {
                    3
                } else {
                    4
                }
            } else {
                5
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
            if 1 {
                1
            } else if 2 {
                2
            } else if 3 {
                3
            } else {
                4
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
}
