use crate::{
    ast,
    lexer::{self, types as lex},
    qombineur::{any, enclosed_within, separated_by, this, Parsed, Parsimonious},
};
use std::marker;

pub mod types {
    use std::result;

    use crate::lexer;

    pub type Result<A> = result::Result<A, Error>;

    #[derive(Debug)]
    pub enum Error {
        Lexical(lexer::types::Error),
        Eof,
    }

    impl From<lexer::types::Error> for Error {
        fn from(value: lexer::types::Error) -> Self {
            Error::Lexical(value)
        }
    }
}

pub fn analyze(input: &[char]) -> types::Result<ast::Program> {
    let toks = lexer::run(input)?;
    match block().parse(&toks) {
        Parsed::Emits(entry_point, remains) => {
            println!("Remains: {remains:#?}");
            Ok(ast::Program {
                definitions: vec![],
                entry_point,
            })
        }
        Parsed::Diverges => Err(types::Error::Eof),
    }
}

#[derive(Clone, Copy)]
struct Thunk<A>(marker::PhantomData<A>);

fn token<F, A>(p: F) -> impl Parsimonious<A, Token = lex::Token>
where
    F: FnOnce(&lex::Token) -> Option<A> + Clone,
    A: Clone,
{
    any::<lex::Token>().filter_map(p)
}

fn literal() -> impl Parsimonious<ast::Expression, Token = lex::Token> {
    token(|tok| match tok {
        lex::Token::Literal(lit) => Some(ast::Expression::Literal(into_constant(lit))),
        _otherwise => None,
    })
}

fn into_constant(lit: &lex::Literal) -> ast::Constant {
    match lit {
        lex::Literal::Text(x) => ast::Constant::Text(x.into()),
        lex::Literal::Integer(x) => ast::Constant::Int(*x),
        lex::Literal::FloatingPoint(x) => ast::Constant::Float(*x),
        lex::Literal::Boolean(x) => ast::Constant::Boolean(*x),
    }
}

fn identifier() -> impl Parsimonious<String, Token = lex::Token> {
    token(|tok| match tok {
        lex::Token::Identifier(lex::Identifier(x)) => Some(x.into()),
        _otherwise => None,
    })
}

fn keyword(k: lex::Keyword) -> impl Parsimonious<(), Token = lex::Token> {
    this(lex::Token::Keyword(k)).ignore()
}

fn separator(s: lex::Separator) -> impl Parsimonious<(), Token = lex::Token> {
    this(lex::Token::Separator(s)).ignore()
}

fn expression() -> Thunk<ast::Expression> {
    Thunk(marker::PhantomData)
}

impl Parsimonious<ast::Expression> for Thunk<ast::Expression> {
    type Token = lex::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, ast::Expression, Self::Token> {
        expression_inner().parse(input)
    }
}

fn expression_inner() -> impl Parsimonious<ast::Expression, Token = lex::Token> {
    let operator = token(|t| match t {
        lex::Token::Separator(lex::Separator::Plus) => Some(ast::Operator::Plus),
        lex::Token::Separator(lex::Separator::Minus) => Some(ast::Operator::Minus),
        _otherwise => None,
    });
    let sequence = operator.and_also(infix_term()).zero_or_more();
    infix_term().and_also(sequence).map(|(lhs, rhss)| {
        rhss.into_iter()
            .fold(lhs, |lhs, (symbol, rhs)| ast::Expression::ApplyInfix {
                lhs: Box::new(lhs),
                symbol,
                rhs: Box::new(rhs),
            })
    })
}

fn expression_rhs() -> impl Parsimonious<ast::Expression, Token = lex::Token> {
    apply().or_else(literal())
}

fn infix_term() -> impl Parsimonious<ast::Expression, Token = lex::Token> {
    let operator = token(|t| match t {
        lex::Token::Separator(lex::Separator::Slash) => Some(ast::Operator::Divides),
        lex::Token::Separator(lex::Separator::Star) => Some(ast::Operator::Times),
        _otherwise => None,
    });
    let sequence = operator.and_also(infix_factor()).zero_or_more();
    infix_factor().and_also(sequence).map(|(lhs, rhss)| {
        rhss.into_iter()
            .fold(lhs, |lhs, (symbol, rhs)| ast::Expression::ApplyInfix {
                lhs: Box::new(lhs),
                symbol,
                rhs: Box::new(rhs),
            })
    })
}

fn infix_factor() -> impl Parsimonious<ast::Expression, Token = lex::Token> {
    enclosed_within(
        separator(lex::Separator::LeftParen),
        separator(lex::Separator::RightParen),
        expression(),
    )
    .or_else(expression_rhs())
}

fn apply() -> impl Parsimonious<ast::Expression, Token = lex::Token> {
    let argument_list = enclosed_within(
        separator(lex::Separator::LeftParen),
        separator(lex::Separator::RightParen),
        separated_by(expression(), separator(lex::Separator::Comma)),
    );

    identifier()
        .and_also(argument_list)
        .map(|(symbol, arguments)| ast::Expression::Apply {
            symbol: ast::Select::Function(ast::Name::simple(&symbol)),
            arguments,
        })
}

fn statement() -> Thunk<ast::Statement> {
    Thunk(marker::PhantomData)
}

impl Parsimonious<ast::Statement> for Thunk<ast::Statement> {
    type Token = lex::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, ast::Statement, Self::Token> {
        statement_inner().parse(input)
    }
}

fn statement_inner() -> impl Parsimonious<ast::Statement, Token = lex::Token> {
    let_statement()
        .or_else(while_statement())
        .or_else(statement_expression())
}

// All these Token = lex::Token have to go. How? Why aren't they "inherited?"
fn let_statement() -> impl Parsimonious<ast::Statement, Token = lex::Token> {
    keyword(lex::Keyword::Let)
        .skip_left(identifier())
        .skip_right(separator(lex::Separator::Assign))
        .and_also(expression())
        .skip_right(separator(lex::Separator::Semicolon))
        .map(|(lhs, rhs)| ast::Statement::Let { lhs, rhs })
}

fn while_statement() -> impl Parsimonious<ast::Statement, Token = lex::Token> {
    keyword(lex::Keyword::While)
        .skip_left(expression())
        .and_also(block())
        .map(|(predicate, body)| ast::Statement::While { predicate, body })
}

fn block() -> impl Parsimonious<ast::Block, Token = lex::Token> {
    enclosed_within(
        separator(lex::Separator::LeftBrace),
        separator(lex::Separator::RightBrace),
        statement().zero_or_more(),
    )
    .map(|statements| ast::Block { statements })
}

fn statement_expression() -> impl Parsimonious<ast::Statement, Token = lex::Token> {
    expression()
        .skip_right(separator(lex::Separator::Semicolon))
        .map(ast::Statement::Expression)
}

#[cfg(test)]
mod tests {
    use crate::ast;

    use super::{
        ast::*,
        lex::Literal::*,
        lex::Separator::*,
        lex::{Token as T, *},
        Parsimonious,
    };

    #[test]
    fn let_statement() {
        let input = &[
            T::Keyword(Keyword::Let),
            T::Identifier(Identifier("quux".into())),
            T::Separator(Assign),
            T::Literal(Integer(427)),
            T::Separator(Semicolon),
        ];
        let was = super::statement().parse(input);
        assert_eq!(
            was.clone().emits(),
            Some(Statement::Let {
                lhs: "quux".into(),
                rhs: Expression::Literal(Constant::Int(427))
            })
        );

        let input = &[
            T::Keyword(Keyword::Let),
            T::Identifier(Identifier("quux".into())),
            T::Separator(Assign),
            T::Literal(Integer(123)),
            T::Separator(Plus),
            T::Literal(Integer(456)),
            T::Separator(Semicolon),
        ];
        let was = super::statement().parse(input);
        assert_eq!(
            was.clone().emits(),
            Some(Statement::Let {
                lhs: "quux".into(),
                rhs: Expression::ApplyInfix {
                    lhs: Box::new(Expression::Literal(Constant::Int(123))),
                    symbol: Operator::Plus,
                    rhs: Box::new(Expression::Literal(Constant::Int(456)))
                }
            })
        );
    }

    #[test]
    fn while_statement() {
        let input = &[
            T::Keyword(Keyword::While),
            T::Literal(Boolean(true)),
            T::Separator(Separator::LeftBrace),
            T::Keyword(Keyword::Let),
            T::Identifier(Identifier("x".into())),
            T::Separator(Assign),
            T::Literal(FloatingPoint(427.427)),
            T::Separator(Semicolon),
            T::Separator(Separator::RightBrace),
        ];
        let was = super::statement().parse(input);
        assert_eq!(
            was.clone().emits(),
            Some(Statement::While {
                predicate: Expression::Literal(Constant::Boolean(true)),
                body: Block {
                    statements: vec![Statement::Let {
                        lhs: "x".into(),
                        rhs: Expression::Literal(Constant::Float(427.427))
                    }]
                }
            })
        );

        let input = &[
            T::Keyword(Keyword::While),
            T::Literal(Literal::Integer(123)),
            T::Separator(Separator::Star), // change to LessThan once implemented
            T::Literal(Literal::Integer(456)),
            T::Separator(Separator::LeftBrace),
            T::Keyword(Keyword::Let),
            T::Identifier(Identifier("x".into())),
            T::Separator(Assign),
            T::Literal(FloatingPoint(427.427)),
            T::Separator(Semicolon),
            T::Identifier(Identifier("foo".into())),
            T::Separator(Separator::LeftParen),
            T::Literal(Literal::Integer(1)),
            T::Separator(Separator::Plus),
            T::Literal(Literal::Integer(2)),
            T::Separator(Separator::RightParen),
            T::Separator(Semicolon),
            T::Separator(Separator::RightBrace),
        ];
        let was = super::statement().parse(input);
        assert_eq!(
            was.clone().emits(),
            Some(Statement::While {
                predicate: Expression::ApplyInfix {
                    lhs: Box::new(Expression::Literal(Constant::Int(123))),
                    symbol: Operator::Times,
                    rhs: Box::new(Expression::Literal(Constant::Int(456)))
                },
                body: Block {
                    statements: vec![
                        Statement::Let {
                            lhs: "x".into(),
                            rhs: Expression::Literal(Constant::Float(427.427))
                        },
                        Statement::Expression(Expression::Apply {
                            symbol: Select::Function(ast::Name::simple("foo")),
                            arguments: vec![ast::Expression::ApplyInfix {
                                lhs: Box::new(Expression::Literal(Constant::Int(1))),
                                symbol: Operator::Plus,
                                rhs: Box::new(Expression::Literal(Constant::Int(2)))
                            }]
                        }),
                    ]
                }
            })
        );
    }

    #[test]
    fn infix_expressions() {
        //        let input = r#"1+2+3-4*5"#;
        let input = &[
            T::Literal(Literal::Integer(1)),
            T::Separator(Separator::Plus),
            T::Literal(Literal::Integer(2)),
            T::Separator(Separator::Plus),
            T::Literal(Literal::Integer(3)),
            T::Separator(Separator::Minus),
            T::Literal(Literal::Integer(4)),
            T::Separator(Separator::Star),
            T::Literal(Literal::Integer(5)),
        ];
        let was = super::expression().parse(input);
        assert_eq!(
            was.clone().emits(),
            Some(Expression::ApplyInfix {
                lhs: Box::new(Expression::ApplyInfix {
                    lhs: Box::new(Expression::ApplyInfix {
                        lhs: Box::new(Expression::Literal(Constant::Int(1))),
                        symbol: Operator::Plus,
                        rhs: Box::new(Expression::Literal(Constant::Int(2)))
                    }),
                    symbol: Operator::Plus,
                    rhs: Box::new(Expression::Literal(Constant::Int(3)))
                }),
                symbol: Operator::Minus,
                rhs: Box::new(Expression::ApplyInfix {
                    lhs: Box::new(Expression::Literal(Constant::Int(4))),
                    symbol: Operator::Times,
                    rhs: Box::new(Expression::Literal(Constant::Int(5)))
                })
            })
        )
    }

    #[test]
    fn apply() {
        // foo(1 + 2) * 3
        let input = &[
            T::Identifier(Identifier("foo".into())),
            T::Separator(Separator::LeftParen),
            T::Literal(Literal::Integer(1)),
            T::Separator(Separator::Plus),
            T::Literal(Literal::Integer(2)),
            T::Separator(Separator::RightParen),
            T::Separator(Separator::Star),
            T::Literal(Literal::Integer(3)),
        ];
        let was = super::expression().parse(input);
        assert_eq!(
            was.clone().emits(),
            Some(ast::Expression::ApplyInfix {
                lhs: Box::new(Expression::Apply {
                    symbol: Select::Function(ast::Name::simple("foo")),
                    arguments: vec![ast::Expression::ApplyInfix {
                        lhs: Box::new(Expression::Literal(Constant::Int(1))),
                        symbol: Operator::Plus,
                        rhs: Box::new(Expression::Literal(Constant::Int(2)))
                    }]
                }),
                symbol: Operator::Times,
                rhs: Box::new(Expression::Literal(Constant::Int(3)))
            })
        )
    }
}
