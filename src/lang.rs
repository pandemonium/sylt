enum Token {
    Identifier(String),
    Literal(Constant),
    Separator(Symbol),
}

enum Constant {
    Integer(i64),
    Decimal(f64),
    Text(String),
    Boolean(bool),
}

enum Symbol {
    LeftParen, RightParen,
    LeftBrace, RightBrace,
    Comma, Semi, Period,
    Assign,
    Plus, Minus, Times, Divides, Modulo,
    Quote, Apostrophe,
}