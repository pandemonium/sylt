use crate::qombineur::*;
use std::result;

type Result<A> = result::Result<A, types::Error>;

pub mod types {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Error {
        Expected(),
        Eof,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum Token {
        Separator(Separator),
        Identifier(Identifier),
        Keyword(Keyword),
        Literal(Literal),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum Literal {
        Text(String),
        Integer(i64),
        FloatingPoint(f64),
        Boolean(bool),
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Separator {
        LeftParen,
        RightParen,
        LeftBrace,
        RightBrace,
        LessThan,
        LessThanOrEqual,
        GreaterThan,
        GreaterThanOrEqual,
        Equals,
        Colon,
        Semicolon,
        Comma,
        Period,
        DoubleQuote,
        SingleQuote,
        Assign,
        Plus,
        Minus,
        Star,
        Slash,
        Percent,
        ThinRightArrow,
    }

    lazy_static! {
        pub static ref SIMPLE_SEPARATORS: &'static [Separator] = &[
            Separator::LeftParen,
            Separator::RightParen,
            Separator::LeftBrace,
            Separator::RightBrace,
            Separator::LessThan,
            Separator::GreaterThan,
            Separator::Colon,
            Separator::Semicolon,
            Separator::Comma,
            Separator::Period,
            Separator::DoubleQuote,
            Separator::SingleQuote,
            Separator::Assign,
            Separator::Plus,
            Separator::Minus,
            Separator::Star,
            Separator::Slash,
            Separator::Percent,
        ];
    }

    lazy_static! {
        pub static ref SIMPLE_SEPARATOR_CHARS: Vec<char> = make_separator_chars();
    }

    fn make_separator_chars() -> Vec<char> {
        SIMPLE_SEPARATORS
            .iter()
            .map(|c| c.into_char())
            .collect::<Vec<_>>()
    }

    impl Separator {
        pub const fn into_char(self) -> char {
            match self {
                Self::LeftParen => '(',
                Self::RightParen => ')',
                Self::LeftBrace => '{',
                Self::RightBrace => '}',
                Self::LessThan => '<',
                Self::GreaterThan => '>',
                Self::Colon => ':',
                Self::Semicolon => ';',
                Self::Comma => ',',
                Self::Period => '.',
                Self::DoubleQuote => '"',
                Self::SingleQuote => '\'',
                Self::Assign => '=',
                Self::Plus => '+',
                Self::Minus => '-',
                Self::Star => '*',
                Self::Slash => '/',
                Self::Percent => '%',

                // Compounds, never match these
                Self::LessThanOrEqual => '$',
                Self::GreaterThanOrEqual => '$',
                Self::Equals => '$',
                Self::ThinRightArrow => '$',
            }
        }

        pub fn try_from_char(c: char) -> Option<Self> {
            SIMPLE_SEPARATORS
                .iter()
                .find(|s| s.into_char() == c)
                .copied()
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct Identifier(pub String);

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Keyword {
        If,
        Else,
        While,
        Let,
        Fn,
        Return,
    }

    impl Keyword {
        pub fn try_from(Identifier(id): Identifier) -> Option<Self> {
            match id.as_str() {
                "if" => Some(Self::If),
                "else" => Some(Self::Else),
                "while" => Some(Self::While),
                "let" => Some(Self::Let),
                "fn" => Some(Self::Fn),
                "return" => Some(Self::Return),
                _otherwise => None,
            }
        }
    }
}

pub mod parsers {
    use super::types::{Identifier, Literal, Token};
    use crate::{
        lexer::types,
        qombineur::{char, digit, enclosed_within, one_of, string, such_that, Parsimonious},
    };

    lazy_static! {
        static ref LEGAL_IDENTIFIER_CHARS: Vec<char> = make_legal_identifier_chars();
    }

    fn make_legal_identifier_chars() -> Vec<char> {
        let mut identifiers = vec![];
        identifiers.extend('a'..='z');
        identifiers.extend('A'..='Z');
        identifiers.push('_');
        identifiers
    }

    fn whitespace() -> impl Parsimonious<(), Token = char> {
        such_that(|c| char::is_whitespace(*c))
            .zero_or_more()
            .ignore()
    }

    fn lexeme<P, A>(produce: P) -> impl Parsimonious<A, Token = P::Token>
    where
        P: Parsimonious<A, Token = char>,
        A: Clone,
    {
        whitespace().skip_left(produce)
    }

    fn compound_separator() -> impl Parsimonious<Token, Token = char> {
        let gte = string(">=").map(|_| types::Separator::GreaterThanOrEqual);
        let lte = string("<=").map(|_| types::Separator::LessThanOrEqual);
        let equals = string("==").map(|_| types::Separator::Equals);
        let arrow = string("->").map(|_| types::Separator::ThinRightArrow);

        gte.or_else(lte)
            .or_else(equals)
            .or_else(arrow)
            .map(types::Token::Separator)
    }

    fn separator() -> impl Parsimonious<Token, Token = char> {
        one_of(&types::SIMPLE_SEPARATOR_CHARS).map(|c| {
            types::Separator::try_from_char(c)
                .map(types::Token::Separator)
                .expect("legal separator char")
        })
    }

    fn literal() -> impl Parsimonious<Token, Token = char> {
        floating_point()
            .or_else(integer())
            .or_else(text())
            .or_else(boolean())
            .map(Token::Literal)
    }

    fn integer() -> impl Parsimonious<Literal, Token = char> {
        digit()
            .one_or_more()
            .map(|image| {
                image
                    .iter()
                    .collect::<String>()
                    .parse::<i64>()
                    .ok()
                    .map(Literal::Integer)
            })
            .filter_map(|x| x.clone())
    }

    fn boolean() -> impl Parsimonious<Literal, Token = char> {
        string("True")
            .map(|_| Literal::Boolean(true))
            .or_else(string("False").map(|_| Literal::Boolean(false)))
    }

    fn floating_point() -> impl Parsimonious<Literal, Token = char> {
        digit()
            .one_or_more()
            .and_also(char('.').skip_left(digit().one_or_more()))
            .map(|(mut image, decimals)| {
                image.push('.');
                image.extend(decimals);

                // Perhaps this FromStr + filter_map combo can be an abstraction?
                image
                    .iter()
                    .collect::<String>()
                    .parse::<f64>()
                    .ok()
                    .map(Literal::FloatingPoint)
            })
            .filter_map(|x| x.clone())
    }

    fn text() -> impl Parsimonious<Literal, Token = char> {
        enclosed_within(
            char('"'),
            char('"'),
            such_that(|c| c != &'"').zero_or_more(),
        )
        .map(|xs| xs.into_iter().collect::<String>())
        .map(Literal::Text)
    }

    fn identifier() -> impl Parsimonious<Identifier, Token = char> {
        one_of(&LEGAL_IDENTIFIER_CHARS)
            .one_or_more()
            .map(|cs| cs.iter().collect::<String>())
            .map(types::Identifier)
    }

    fn identifier_or_keyword() -> impl Parsimonious<Token, Token = char> {
        identifier().map(|id| {
            types::Keyword::try_from(id.clone())
                .map(types::Token::Keyword)
                .unwrap_or_else(|| types::Token::Identifier(id))
        })
    }

    pub fn token() -> impl Parsimonious<Token, Token = char> {
        lexeme(
            literal()
                .or_else(identifier_or_keyword())
                .or_else(compound_separator())
                .or_else(separator()),
        )
    }
}

// Positions? Should this borrow from the source?
pub fn run(source: &[char]) -> Result<Vec<types::Token>> {
    match parsers::token().one_or_more().parse(source) {
        Parsed::Emits(toks, ..) => Ok(toks),

        // What to do with this?
        //   I have no error reporting.
        Parsed::Diverges => Err(types::Error::Eof),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        run,
        types::{Token as T, *},
    };

    #[test]
    fn let_statement() {
        let input = "let quux = foo;";
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Keyword(Keyword::Let),
                T::Identifier(Identifier("quux".into())),
                T::Separator(Separator::Assign),
                T::Identifier(Identifier("foo".into())),
                T::Separator(Separator::Semicolon),
            ]
        );

        let input = "let quux = 427;";
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Keyword(Keyword::Let),
                T::Identifier(Identifier("quux".into())),
                T::Separator(Separator::Assign),
                T::Literal(Literal::Integer(427)),
                T::Separator(Separator::Semicolon),
            ]
        );

        let input = "let quux = -427.314;";
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Keyword(Keyword::Let),
                T::Identifier(Identifier("quux".into())),
                T::Separator(Separator::Assign),
                T::Separator(Separator::Minus),
                T::Literal(Literal::FloatingPoint(427.314)),
                T::Separator(Separator::Semicolon),
            ]
        );

        let input = r#"let quux = "427.314";"#;
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Keyword(Keyword::Let),
                T::Identifier(Identifier("quux".into())),
                T::Separator(Separator::Assign),
                T::Literal(Literal::Text("427.314".into())),
                T::Separator(Separator::Semicolon),
            ]
        );
    }

    #[test]
    fn tokenör() {
        let input = r#"True / "427.314;" + println("{}", -427); "#;
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Literal(Literal::Boolean(true)),
                T::Separator(Separator::Slash),
                T::Literal(Literal::Text("427.314;".into())),
                T::Separator(Separator::Plus),
                T::Identifier(Identifier("println".into())),
                T::Separator(Separator::LeftParen),
                T::Literal(Literal::Text("{}".into())),
                T::Separator(Separator::Comma),
                T::Separator(Separator::Minus),
                T::Literal(Literal::Integer(427)),
                T::Separator(Separator::RightParen),
                T::Separator(Separator::Semicolon),
            ]
        );
    }

    #[test]
    fn infix_expressions() {
        let input = r#"1+2+3-4*5"#;
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Literal(Literal::Integer(1)),
                T::Separator(Separator::Plus),
                T::Literal(Literal::Integer(2)),
                T::Separator(Separator::Plus),
                T::Literal(Literal::Integer(3)),
                T::Separator(Separator::Minus),
                T::Literal(Literal::Integer(4)),
                T::Separator(Separator::Star),
                T::Literal(Literal::Integer(5)),
            ]
        );
    }

    #[test]
    fn apply() {
        let input = r#" foo ( 1 + 2 ) * 3"#;
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Identifier(Identifier("foo".into())),
                T::Separator(Separator::LeftParen),
                T::Literal(Literal::Integer(1)),
                T::Separator(Separator::Plus),
                T::Literal(Literal::Integer(2)),
                T::Separator(Separator::RightParen),
                T::Separator(Separator::Star),
                T::Literal(Literal::Integer(3)),
            ]
        );
    }

    #[test]
    fn while_block() {
        let input = r#" while  123*456{  let x=427.427;foo(1+ 2) ;  } "#;
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Keyword(Keyword::While),
                T::Literal(Literal::Integer(123)),
                T::Separator(Separator::Star), // change to LessThan once implemented
                T::Literal(Literal::Integer(456)),
                T::Separator(Separator::LeftBrace),
                T::Keyword(Keyword::Let),
                T::Identifier(Identifier("x".into())),
                T::Separator(Separator::Assign),
                T::Literal(Literal::FloatingPoint(427.427)),
                T::Separator(Separator::Semicolon),
                T::Identifier(Identifier("foo".into())),
                T::Separator(Separator::LeftParen),
                T::Literal(Literal::Integer(1)),
                T::Separator(Separator::Plus),
                T::Literal(Literal::Integer(2)),
                T::Separator(Separator::RightParen),
                T::Separator(Separator::Semicolon),
                T::Separator(Separator::RightBrace),
            ]
        );
    }

    #[test]
    fn compound_separators() {
        let input = r#" fn foo() -> Int { while 2 >= 1 {} while 2 <= 1 {} } "#;
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Keyword(Keyword::Fn),
                T::Identifier(Identifier("foo".into())),
                T::Separator(Separator::LeftParen),
                T::Separator(Separator::RightParen),

                T::Separator(Separator::ThinRightArrow),

                T::Identifier(Identifier("Int".into())),

                T::Separator(Separator::LeftBrace),

                T::Keyword(Keyword::While),
                T::Literal(Literal::Integer(2)),
                T::Separator(Separator::GreaterThanOrEqual),
                T::Literal(Literal::Integer(1)),
                T::Separator(Separator::LeftBrace),
                T::Separator(Separator::RightBrace),

                T::Keyword(Keyword::While),
                T::Literal(Literal::Integer(2)),
                T::Separator(Separator::LessThanOrEqual),
                T::Literal(Literal::Integer(1)),
                T::Separator(Separator::LeftBrace),
                T::Separator(Separator::RightBrace),

                T::Separator(Separator::RightBrace),

            ]
        );
    }
}
