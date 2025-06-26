use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Span<T = usize, C = ()> {
    /// The start position of the span in the source code
    pub start: T,
    /// The end position of the span in the source code
    pub end: T,
    /// Additional context for the span, if any
    pub context: C,
}

/// Expression
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Expr<'a> {
    /// An integer literal
    IntLit {
        /// The integer value
        value: i64,
        /// The span of the literal in the source code
        span: Span,
    },
    /// A boolean literal
    BoolLit {
        /// The boolean value
        value: bool,
        /// The span of the literal in the source code
        span: Span,
    },
    /// A binary operation
    BinOp {
        /// The left-hand side expression
        lhs: Box<Expr<'a>>,
        /// The operator
        op: BinOp,
        /// The right-hand side expression
        rhs: Box<Expr<'a>>,
        /// The span of the expression in the source code
        span: Span,
    },
    /// A unary operation
    UnaryOp {
        /// The operator
        op: UnaryOp,
        /// The expression
        expr: Box<Expr<'a>>,
        /// The span of the expression in the source code
        span: Span,
    },
    /// A function call
    FnCall {
        /// The function name
        name: &'a str,
        /// The arguments
        args: Vec<Expr<'a>>,
        /// The span of the function call in the source code
        span: Span,
    },
    /// A variable reference (identifier)
    VarRef {
        /// The variable name
        name: &'a str,
        /// The span of the variable reference in the source code
        span: Span,
    },
}

/// Binary operator
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum BinOp {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Sub,
    /// Multiplication (*)
    Mul,
    /// Division (/)
    Div,
    /// Equality (==)
    Equal,
    /// Inequality (!=)
    NotEqual,
    /// Less than (<)
    LessThan,
    /// Less than or equal (<=)
    LessThanOrEqual,
    /// Greater than (>)
    GreaterThan,
    /// Greater than or equal (>=)
    GreaterThanOrEqual,
    /// Logical AND (&&)
    And,
    /// Logical OR (||)
    Or,
}

/// Unary operator
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum UnaryOp {
    /// Negation (-)
    Neg,
    /// Logical NOT (!)
    Not,
}

/// Type
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Type<'a> {
    Bool,
    I32,
    I64,
    F32,
    F64,
    Void,
    String,
    Fn {
        /// The return type of the function
        return_type: Box<Type<'a>>,
        /// The parameters of the function
        params: Vec<FnParam<'a, Type<'a>>>,
    },
}

/// Function parameter
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FnParam<'a, Ty = Option<Type<'a>>> {
    /// The name of the parameter
    pub name: &'a str,

    /// The type of the parameter
    pub r#type: Ty,

    /// The span of the parameter in the source code
    pub span: Span,
}

/// Statements
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Stmt<'a, Ty = Option<Type<'a>>> {
    /// A function declaration
    FnDecl {
        /// The name of the function
        name: &'a str,
        /// The parameters of the function
        params: Vec<FnParam<'a, Ty>>,
        /// The return type of the function
        r#type: Ty,
        /// The body of the function
        body: Vec<Stmt<'a, Ty>>,
        /// The span of the function declaration in the source code
        span: Span,
    },

    /// A variable declaration (let)
    LetDecl {
        /// The variable name
        name: &'a str,
        /// The type
        r#type: Ty,
        /// The value
        value: Option<Expr<'a>>,
        /// The span of the variable declaration in the source code
        span: Span,
    },

    /// A mutable variable declaration (var)
    VarDecl {
        /// The variable name
        name: &'a str,
        /// The type
        r#type: Ty,
        /// The value
        value: Option<Expr<'a>>,
        /// The span of the variable declaration in the source code
        span: Span,
    },

    /// An assignment statement
    Assign {
        /// The variable name
        name: &'a str,
        /// The value to assign
        value: Box<Expr<'a>>,
        /// The span of the assignment in the source code
        span: Span,
    },

    /// An if statement
    If {
        /// The condition
        condition: Box<Expr<'a>>,
        /// The then branch
        then_branch: Vec<Stmt<'a, Ty>>,
        /// The else branch
        else_branch: Option<Vec<Stmt<'a, Ty>>>,
        /// The span of the if statement in the source code
        span: Span,
    },

    /// A return statement
    Return {
        /// The expression to return
        expr: Option<Box<Expr<'a>>>,
        /// The span of the return statement in the source code
        span: Span,
    },

    /// An expression statement
    #[allow(clippy::enum_variant_names)]
    ExprStmt {
        /// The expression
        expr: Box<Expr<'a>>,
        /// The span of the expression statement in the source code
        span: Span,
    },

    /// An expression
    Expr {
        /// The expression
        expr: Box<Expr<'a>>,
        /// The span of the expression in the source code
        span: Span,
    },
}

/// Function declaration (for file-level declarations only)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FnDecl<'a, Ty = Option<Type<'a>>> {
    /// The name of the function
    pub name: &'a str,
    /// The parameters of the function
    pub params: Vec<FnParam<'a, Ty>>,
    /// The return type of the function
    pub r#type: Ty,
    /// The body of the function
    pub body: Vec<Stmt<'a, Ty>>,
    /// The span of the function declaration in the source code
    pub span: Span,
}

/// The top-level program structure
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program<'a, Ty = Option<Type<'a>>> {
    /// The function declarations that make up the program
    pub functions: Vec<FnDecl<'a, Ty>>,
}

impl std::fmt::Display for Type<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Bool => write!(f, "bool"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
            Type::Void => write!(f, "void"),
            Type::String => write!(f, "string"),
            Type::Fn {
                return_type,
                params,
            } => {
                write!(f, "fn(")?;
                for (i, param) in params.iter().enumerate() {
                    write!(f, "{}", param)?;
                    if i < params.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ") -> {}", return_type)
            }
        }
    }
}

impl From<chumsky::span::SimpleSpan> for Span {
    fn from(span: chumsky::span::SimpleSpan) -> Self {
        Span {
            start: span.start,
            end: span.end,
            context: span.context,
        }
    }
}

impl Span {
    pub fn to_range(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

impl std::fmt::Display for FnParam<'_, Option<Type<'_>>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.r#type {
            Some(ty) => write!(f, "{}: {}", self.name, ty),
            None => write!(f, "{}", self.name),
        }
    }
}

impl std::fmt::Display for FnParam<'_, Type<'_>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.r#type)
    }
}

impl<'a> Expr<'a> {
    pub fn span(&self) -> &Span {
        match self {
            Expr::IntLit { span, .. } => span,
            Expr::BoolLit { span, .. } => span,
            Expr::BinOp { span, .. } => span,
            Expr::UnaryOp { span, .. } => span,
            Expr::FnCall { span, .. } => span,
            Expr::VarRef { span, .. } => span,
        }
    }
}

impl<'a> Stmt<'a, Type<'a>> {
    #[allow(dead_code)]
    pub fn span(&self) -> &Span {
        match self {
            Stmt::FnDecl { span, .. } => span,
            Stmt::LetDecl { span, .. } => span,
            Stmt::VarDecl { span, .. } => span,
            Stmt::Assign { span, .. } => span,
            Stmt::If { span, .. } => span,
            Stmt::Return { span, .. } => span,
            Stmt::ExprStmt { span, .. } => span,
            Stmt::Expr { span, .. } => span,
        }
    }
}

impl<'a> FnDecl<'a, Type<'a>> {
    #[allow(dead_code)]
    pub fn span(&self) -> &Span {
        &self.span
    }
}
