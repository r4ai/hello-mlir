use logos::Logos;

#[derive(Logos, Clone, PartialEq, Debug)]
pub enum Token<'a> {
    Error,

    #[token("fn")]
    FunctionDeclaration,

    #[token("let")]
    LetDeclaration,

    #[token("var")]
    VarDeclaration,

    #[token("return")]
    Return,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier(&'a str),

    #[regex(r"[0-9]+")]
    Integer(&'a str),

    #[regex(r"[0-9]+\.[0-9]+")]
    Float(&'a str),

    #[regex(r"[0-9]+_i32")]
    TypedIntegerI32(&'a str),

    #[regex(r"[0-9]+_i64")]
    TypedIntegerI64(&'a str),

    #[regex(r"[0-9]+\.[0-9]+_f32")]
    TypedFloatF32(&'a str),

    #[regex(r"[0-9]+\.[0-9]+_f64")]
    TypedFloatF64(&'a str),

    #[token("+")]
    Add,

    #[token("-")]
    Sub,

    #[token("*")]
    Mul,

    #[token("/")]
    Div,

    #[token("==")]
    Equal,

    #[token("!=")]
    NotEqual,

    #[token("<")]
    LessThan,

    #[token("<=")]
    LessThanOrEqual,

    #[token(">")]
    GreaterThan,

    #[token(">=")]
    GreaterThanOrEqual,

    #[token("&&")]
    And,

    #[token("||")]
    Or,

    #[token("!")]
    Not,

    #[token(",")]
    Comma,

    #[token("->")]
    RightArrow,

    #[token(":")]
    Colon,

    #[token(";")]
    Semicolon,

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,

    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    #[token("=")]
    Assign,

    #[regex(r"[ \t\f\n]+", logos::skip)]
    Whitespace,

    // Skip line comments (// ...)
    #[regex(r"//.*", logos::skip)]
    LineComment,

    // Skip block comments (/* ... */)
    #[regex(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", logos::skip)]
    BlockComment,
}

impl std::fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FunctionDeclaration => write!(f, "fn"),
            Self::LetDeclaration => write!(f, "let"),
            Self::VarDeclaration => write!(f, "var"),
            Self::Return => write!(f, "return"),
            Self::If => write!(f, "if"),
            Self::Else => write!(f, "else"),
            Self::Identifier(value) => write!(f, "{value}"),
            Self::Integer(value) => write!(f, "{value}"),
            Self::Float(value) => write!(f, "{value}"),
            Self::TypedIntegerI32(value) => write!(f, "{value}"),
            Self::TypedIntegerI64(value) => write!(f, "{value}"),
            Self::TypedFloatF32(value) => write!(f, "{value}"),
            Self::TypedFloatF64(value) => write!(f, "{value}"),
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Equal => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::LessThan => write!(f, "<"),
            Self::LessThanOrEqual => write!(f, "<="),
            Self::GreaterThan => write!(f, ">"),
            Self::GreaterThanOrEqual => write!(f, ">="),
            Self::And => write!(f, "&&"),
            Self::Or => write!(f, "||"),
            Self::Not => write!(f, "!"),
            Self::Comma => write!(f, ","),
            Self::Colon => write!(f, ":"),
            Self::Semicolon => write!(f, ";"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::RightArrow => write!(f, "->"),
            Self::Assign => write!(f, "="),
            Self::Whitespace => write!(f, "<whitespace>"),
            Self::LineComment => write!(f, "<line_comment>"),
            Self::BlockComment => write!(f, "<block_comment>"),
            Self::Error => write!(f, "<e>"),
        }
    }
}
