use serde::Serialize;

/// Expression
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Expr<'a> {
    /// An integer literal
    IntLit(i64),
    /// A boolean literal
    BoolLit(bool),
    /// A binary operation
    BinOp {
        /// The left-hand side expression
        lhs: Box<Expr<'a>>,
        /// The operator
        op: BinOp,
        /// The right-hand side expression
        rhs: Box<Expr<'a>>,
    },
    /// A unary operation
    UnaryOp {
        /// The operator
        op: UnaryOp,
        /// The expression
        expr: Box<Expr<'a>>,
    },
    /// A function call
    FnCall {
        /// The function name
        name: &'a str,
        /// The arguments
        args: Vec<Expr<'a>>,
    },
    /// A variable reference (identifier)
    VarRef {
        /// The variable name
        name: &'a str,
    },
}

/// Binary operator
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum UnaryOp {
    /// Negation (-)
    Neg,
    /// Logical NOT (!)
    Not,
}

/// Type
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
    Void,
    String,
}

/// Function parameter
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FnParam<'a, Ty = Option<Type>> {
    /// The name of the parameter
    pub name: &'a str,

    /// The type of the parameter
    pub r#type: Ty,
}

pub type FnParamTyped<'a> = FnParam<'a, Type>;

/// Statements
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Stmt<'a, Ty = Option<Type>> {
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
    },

    /// A variable declaration (let)
    LetDecl {
        /// The variable name
        name: &'a str,
        /// The type
        r#type: Ty,
        /// The value
        value: Option<Expr<'a>>,
    },

    /// A mutable variable declaration (var)
    VarDecl {
        /// The variable name
        name: &'a str,
        /// The type
        r#type: Ty,
        /// The value
        value: Option<Expr<'a>>,
    },

    /// An assignment statement
    Assign {
        /// The variable name
        name: &'a str,
        /// The value to assign
        value: Box<Expr<'a>>,
    },

    /// An if statement
    If {
        /// The condition
        condition: Box<Expr<'a>>,
        /// The then branch
        then_branch: Vec<Stmt<'a, Ty>>,
        /// The else branch
        else_branch: Option<Vec<Stmt<'a, Ty>>>,
    },

    /// A return statement
    Return {
        /// The expression to return
        expr: Option<Box<Expr<'a>>>,
    },

    /// An expression statement
    #[allow(clippy::enum_variant_names)]
    ExprStmt {
        /// The expression
        expr: Box<Expr<'a>>,
    },

    /// An expression
    Expr {
        /// The expression
        expr: Box<Expr<'a>>,
    },
}

pub type StmtTyped<'a> = Stmt<'a, Type>;

/// The top-level program structure
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program<'a, Ty = Option<Type>> {
    /// The expression that makes up the program
    pub stmts: Vec<Stmt<'a, Ty>>,
}

pub type ProgramTyped<'a> = Program<'a, Type>;
