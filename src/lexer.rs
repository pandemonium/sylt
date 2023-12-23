use self::types::Token;
use crate::kombi::{ParseResult, ParseState, Parser};
use std::result;

type Result<A> = result::Result<A, types::Error>;

pub mod types {
    use crate::kombi::Position;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Error {
        Expected(Position),
        Eof,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum Token {
        Separator(Separator),
        Identifier(Identifier),
        Keyword(Keyword),
        Literal(Literal),
    }

    impl Token {
        fn _token_type(&self) -> TokenType {
            match self {
                Token::Separator(..) => TokenType::Separator,
                Token::Identifier(..) => TokenType::Identifier,
                Token::Keyword(..) => TokenType::Keyword,
                Token::Literal(..) => TokenType::Literal,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum TokenType {
        Separator,
        Identifier,
        Keyword,
        Literal,
    }

    pub struct Lexeme {
        _token: Token,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum Literal {
        Text(String),
        Integer(i64),
        FloatingPoint(f64),
        Boolean(bool),
    }

    //OnceLock

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Separator {
        LeftParen,
        RightParen,
        LeftBrace,
        RightBrace,
        LeftBracket,
        RightBracket,
        LessThan,
        LessThanOrEqual,
        GreaterThan,
        GreaterThanOrEqual,
        Equals,
        NotEqual,
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

    impl Separator {
        pub const fn into_char(self) -> char {
            match self {
                Self::LeftParen => '(',
                Self::RightParen => ')',
                Self::LeftBrace => '{',
                Self::RightBrace => '}',
                Self::LeftBracket => ']',
                Self::RightBracket => '[',
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
                Self::NotEqual => '$',
                Self::ThinRightArrow => '$',
            }
        }

        pub fn try_from_char(c: char) -> Option<Self> {
            match c {
                '(' => Some(Self::LeftParen),
                ')' => Some(Self::RightParen),
                '{' => Some(Self::LeftBrace),
                '}' => Some(Self::RightBrace),
                '[' => Some(Self::LeftBracket),
                ']' => Some(Self::RightBracket),
                '<' => Some(Self::LessThan),
                '>' => Some(Self::GreaterThan),
                ':' => Some(Self::Colon),
                ';' => Some(Self::Semicolon),
                ',' => Some(Self::Comma),
                '.' => Some(Self::Period),
                '"' => Some(Self::DoubleQuote),
                '\'' => Some(Self::SingleQuote),
                '=' => Some(Self::Assign),
                '+' => Some(Self::Plus),
                '-' => Some(Self::Minus),
                '*' => Some(Self::Star),
                '/' => Some(Self::Slash),
                '%' => Some(Self::Percent),
                _otherwise => None,
            }
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
        And,
        Or,
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
                "and" => Some(Self::And),
                "or" => Some(Self::Or),
                _otherwise => None,
            }
        }
    }
}

mod kombi_parsers {
    use super::types::{self, Identifier, Keyword, Literal, Separator, Token};
    use crate::{
        expect,
        kombi::{enclosed_within, filter_map, string, such_that, Parser, Positioned},
    };

    fn whitespace() -> impl Parser<In = char, Out = Vec<char>> {
        such_that(|c| char::is_whitespace(*c))
            .with_positions()
            .zero_or_more()
    }

    fn lexeme<P>(produce: P) -> impl Parser<In = char, Out = P::Out>
    where
        P: Parser<In = char>,
        P::Out: Clone,
    {
        whitespace().skip_left(produce)
    }

    fn separator() -> impl Parser<In = char, Out = Token> {
        filter_map(|c| types::Separator::try_from_char(*c))
            .with_positions()
            .map(Token::Separator)
    }

    fn compound_separator() -> impl Parser<In = char, Out = Token> {
        let gte = string(">=").map(|_| Separator::GreaterThanOrEqual);
        let lte = string("<=").map(|_| Separator::LessThanOrEqual);
        let equals = string("==").map(|_| Separator::Equals);
        let not_equal = string("!=").map(|_| Separator::NotEqual);
        let arrow = string("->").map(|_| Separator::ThinRightArrow);

        gte.or_else(lte)
            .or_else(equals)
            .or_else(not_equal)
            .or_else(arrow)
            .map(Token::Separator)
    }

    fn literal() -> impl Parser<In = char, Out = Token> {
        floating_point()
            .or_else(text())
            .or_else(boolean())
            .or_else(integer())
            .map(Token::Literal)
    }

    fn integer() -> impl Parser<In = char, Out = Literal> {
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

    fn digit() -> impl Parser<In = char, Out = char> {
        such_that(|c| char::is_digit(*c, 10)).with_positions()
    }

    fn boolean() -> impl Parser<In = char, Out = Literal> {
        string("True")
            .map(|_| Literal::Boolean(true))
            .or_else(string("False").map(|_| Literal::Boolean(false)))
    }

    fn floating_point() -> impl Parser<In = char, Out = Literal> {
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

    fn text() -> impl Parser<In = char, Out = Literal> {
        enclosed_within(
            char('"'),
            char('"'),
            such_that(|c| c != &'"').zero_or_more(),
        )
        .map(|xs| xs.into_iter().collect::<String>())
        .map(Literal::Text)
    }

    fn char(c: char) -> impl Parser<In = char, Out = char> {
        such_that(move |x| x == &c).with_positions()
    }

    fn identifier() -> impl Parser<In = char, Out = Identifier> {
        expect!(ch @ ('a'..='z' | 'A'..='Z' | '_') => *ch)
            .with_positions()
            .one_or_more()
            .map(|cs| cs.iter().collect::<String>())
            .map(Identifier)
    }

    fn identifier_or_keyword() -> impl Parser<In = char, Out = Token> {
        identifier().map(|id| {
            Keyword::try_from(id.clone())
                .map(Token::Keyword)
                .unwrap_or_else(|| Token::Identifier(id))
        })
    }

    pub fn token() -> impl Parser<In = char, Out = Token> {
        lexeme(
            literal()
                .or_else(identifier_or_keyword())
                .or_else(compound_separator())
                .or_else(separator()),
        )
    }
}

pub fn run(input: &[char]) -> Result<Vec<Token>> {
    let result = kombi_parsers::token()
        .one_or_more()
        .parse(ParseState::new(input));
    let ParseResult { state, parsed } = result;
    println!("Pos: {:?}", state.at);
    if let Some(toks) = parsed {
        Ok(toks)
    } else {
        Err(types::Error::Expected(state.at))
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
    fn array_declaration() {
        let input = r#" let xs =[42;  427 ] ;   "#;
        let was = run(&input.chars().collect::<Vec<_>>());
        assert_eq!(
            was.unwrap(),
            vec![
                T::Keyword(Keyword::Let),
                T::Identifier(Identifier("xs".into())),
                T::Separator(Separator::Assign),
                T::Separator(Separator::LeftBracket),
                T::Literal(Literal::Integer(42)),
                T::Separator(Separator::Semicolon),
                T::Literal(Literal::Integer(427)),
                T::Separator(Separator::RightBracket),
                T::Separator(Separator::Semicolon),
            ]
        );
    }

    #[test]
    fn compound_separators() {
        let input = "fn foo() -> Int {while 2 >= 1 {} while  2 <= 1 {}}";
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

    #[test]
    fn spaces() {
        let input = "



        "
        .chars()
        .collect::<Vec<_>>();

        for c in input {
            print!("{},", c as i32)
        }
    }
}
