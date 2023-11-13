use std::marker;

use crate::{
    ast, expect,
    kombi::{enclosed_within, filter_map, separated_by, ParseResult, ParseState, Parser},
    lexer::{self, types as lex},
};

pub mod types {
    use crate::{kombi, lexer};
    use std::result;

    pub type Result<A> = result::Result<A, Error>;

    #[derive(Debug)]
    pub enum Error {
        Lexical(lexer::types::Error),
        Parse(ParseInfo),
        Eof,
    }

    #[derive(Debug)]
    pub struct ParseInfo {
        location: kombi::Position,
        last_token: Vec<lexer::types::Token>,
        remains: Vec<lexer::types::Token>,
    }

    impl From<lexer::types::Error> for Error {
        fn from(value: lexer::types::Error) -> Self {
            Error::Lexical(value)
        }
    }

    impl<'a> From<kombi::ParseState<'a, lexer::types::Token>> for ParseInfo {
        fn from(value: kombi::ParseState<'a, lexer::types::Token>) -> Self {
            Self {
                location: value.at,
                last_token: value.token.to_vec(),
                remains: value.remains.to_vec(),
            }
        }
    }
}

pub fn analyze(input: &[char]) -> types::Result<ast::Program> {
    let toks = lexer::run(input)?;
    let ParseResult { state, parsed } = program_declaration().parse(ParseState::new(&toks));

    parsed.ok_or_else(|| types::Error::Parse(state.into()))
}

#[derive(Clone, Copy)]
struct Thunk<A>(marker::PhantomData<A>);

fn literal() -> impl Parser<In = lex::Token, Out = ast::Expression> {
    expect!(lex::Token::Literal(lit) => into_constant(&lit)).map(ast::Expression::Literal)
}

fn into_constant(lit: &lex::Literal) -> ast::Constant {
    match lit {
        lex::Literal::Text(x) => ast::Constant::Text(x.into()),
        lex::Literal::Integer(x) => ast::Constant::Int(*x),
        lex::Literal::FloatingPoint(x) => ast::Constant::Float(*x),
        lex::Literal::Boolean(x) => ast::Constant::Boolean(*x),
    }
}

fn identifier() -> impl Parser<In = lex::Token, Out = String> {
    expect!(lex::Token::Identifier(lex::Identifier(s)) => s.into())
}

fn keyword(expected: lex::Keyword) -> impl Parser<In = lex::Token, Out = ()> {
    expect!(lex::Token::Keyword(k) if k == &expected => ())
}

fn separator(expected: lex::Separator) -> impl Parser<In = lex::Token, Out = ()> {
    expect!(lex::Token::Separator(s) if s == &expected => ())
}

fn expression() -> Thunk<ast::Expression> {
    Thunk(marker::PhantomData)
}

impl Parser for Thunk<ast::Expression> {
    type In = lex::Token;
    type Out = ast::Expression;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        expression_inner().parse(input)
    }
}

fn expression_inner() -> impl Parser<In = lex::Token, Out = ast::Expression> {
    let plus = separator(lex::Separator::Plus).map(|_| ast::Operator::Plus);
    let minus = separator(lex::Separator::Minus).map(|_| ast::Operator::Minus);
    let operator = plus.or_else(minus);

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

fn infix_term() -> impl Parser<In = lex::Token, Out = ast::Expression> {
    let times = separator(lex::Separator::Star).map(|_| ast::Operator::Times);
    let divides = separator(lex::Separator::Slash).map(|_| ast::Operator::Divides);
    let operator = times.or_else(divides);

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

fn infix_factor() -> impl Parser<In = lex::Token, Out = ast::Expression> {
    enclosed_within(
        separator(lex::Separator::LeftParen),
        separator(lex::Separator::RightParen),
        expression(),
    )
    .or_else(expression_rhs())
}

fn expression_rhs() -> impl Parser<In = lex::Token, Out = ast::Expression> {
    apply().or_else(literal()).or_else(variable())
}

fn apply() -> impl Parser<In = lex::Token, Out = ast::Expression> {
    let argument_list = enclosed_within(
        separator(lex::Separator::LeftParen),
        separator(lex::Separator::RightParen),
        separated_by(expression(), separator(lex::Separator::Comma)),
    );

    identifier()
        .and_also(argument_list)
        .map(|(symbol, arguments)| ast::Expression::Apply {
            symbol: ast::Select::Function(ast::Name::simple(symbol.as_str())),
            arguments,
        })
}

fn variable() -> impl Parser<In = lex::Token, Out = ast::Expression> {
    identifier().map(|id| ast::Expression::Variable(ast::Name::simple(&id)))
}

fn statement() -> Thunk<ast::Statement> {
    Thunk(marker::PhantomData)
}

impl Parser for Thunk<ast::Statement> {
    type In = lex::Token;
    type Out = ast::Statement;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        statement_inner().parse(input)
    }
}

fn statement_inner() -> impl Parser<In = lex::Token, Out = ast::Statement> {
    let_statement()
        .or_else(while_statement())
        .or_else(statement_expression())
        .or_else(if_statement())
        .or_else(return_statement())
}

// All these Token = lex::Token have to go. How? Why aren't they "inherited?"
fn let_statement() -> impl Parser<In = lex::Token, Out = ast::Statement> {
    keyword(lex::Keyword::Let)
        .skip_left(identifier())
        .skip_right(separator(lex::Separator::Assign))
        .and_also(expression())
        .skip_right(separator(lex::Separator::Semicolon))
        .map(|(lhs, rhs)| ast::Statement::Let { lhs, rhs })
}

fn while_statement() -> impl Parser<In = lex::Token, Out = ast::Statement> {
    keyword(lex::Keyword::While)
        .skip_left(expression())
        .and_also(block())
        .map(|(predicate, body)| ast::Statement::While { predicate, body })
}

fn if_statement() -> impl Parser<In = lex::Token, Out = ast::Statement> {
    keyword(lex::Keyword::If)
        .skip_left(expression())
        .and_also(block())
        .and_also(keyword(lex::Keyword::Else).skip_left(block()))
        .map(|((predicate, when_true), when_false)| ast::Statement::If {
            predicate,
            when_true,
            when_false,
        })
}

fn return_statement() -> impl Parser<In = lex::Token, Out = ast::Statement> {
    keyword(lex::Keyword::Return)
        .skip_left(expression())
        .skip_right(separator(lex::Separator::Semicolon))
        .map(ast::Statement::Return)
}

fn block() -> impl Parser<In = lex::Token, Out = ast::Block> {
    enclosed_within(
        separator(lex::Separator::LeftBrace),
        separator(lex::Separator::RightBrace),
        statement().zero_or_more(),
    )
    .map(|statements| ast::Block { statements })
}

fn statement_expression() -> impl Parser<In = lex::Token, Out = ast::Statement> {
    expression()
        .skip_right(separator(lex::Separator::Semicolon))
        .map(ast::Statement::Expression)
}

fn function_declaration() -> impl Parser<In = lex::Token, Out = ast::Declaration> {
    let parameter = identifier()
        .skip_right(separator(lex::Separator::Colon))
        .and_also(identifier())
        .map(|(name, type_)| ast::Parameter {
            name,
            type_: ast::Type::named(&ast::Name::intrinsic(&type_)),
        });

    let formal_parameters = enclosed_within(
        separator(lex::Separator::LeftParen),
        separator(lex::Separator::RightParen),
        separated_by(parameter, separator(lex::Separator::Comma)),
    );

    keyword(lex::Keyword::Fn)
        .skip_left(identifier().map(|x| ast::Name::simple(&x)))
        .and_also(formal_parameters)
        .skip_right(separator(lex::Separator::ThinRightArrow))
        .and_also(identifier().map(|x| ast::Type::named(&ast::Name::intrinsic(&x))))
        .and_also(block())
        .map(
            |(((name, parameters), return_type), body)| ast::FunctionDef {
                name,
                parameters,
                return_type,
                body,
            },
        )
        .map(ast::Declaration::Function)
}

// I would like to be able to sprinkle statements in between function declarations
// and also not have to enclose the entry point code within braces.
fn program_declaration() -> impl Parser<In = lex::Token, Out = ast::Program> {
    function_declaration()
        .zero_or_more()
        .and_also(block())
        .map(|(definitions, entry_point)| ast::Program {
            declarations: definitions,
            entry_point,
        })
}

#[cfg(test)]
mod tests {
    use super::{
        ast::*,
        lex::Literal::*,
        lex::Separator::*,
        lex::{Token as T, *},
    };
    use crate::{
        ast,
        kombi::{ParseState, Parser},
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
        let was = super::statement().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
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
        let was = super::statement().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
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
        let was = super::statement().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
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
        let was = super::statement().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
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
    fn if_statement() {
        let input = &[
            T::Keyword(Keyword::If),
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
            T::Keyword(Keyword::Else),
            T::Separator(Separator::LeftBrace),
            T::Keyword(Keyword::Let),
            T::Identifier(Identifier("x".into())),
            T::Separator(Assign),
            T::Literal(Literal::Text("Hi, mom".into())),
            T::Separator(Semicolon),
            T::Separator(Separator::RightBrace),
        ];
        let was = super::statement().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
            Some(ast::Statement::If {
                predicate: ast::Expression::ApplyInfix {
                    lhs: Box::new(Expression::Literal(Constant::Int(123))),
                    symbol: Operator::Times,
                    rhs: Box::new(Expression::Literal(Constant::Int(456))),
                },
                when_true: ast::Block {
                    statements: vec![
                        ast::Statement::Let {
                            lhs: "x".into(),
                            rhs: ast::Expression::Literal(Constant::Float(427.427))
                        },
                        Statement::Expression(Expression::Apply {
                            symbol: Select::Function(ast::Name::simple("foo")),
                            arguments: vec![ast::Expression::ApplyInfix {
                                lhs: Box::new(Expression::Literal(Constant::Int(1))),
                                symbol: Operator::Plus,
                                rhs: Box::new(Expression::Literal(Constant::Int(2)))
                            }]
                        })
                    ]
                },
                when_false: ast::Block {
                    statements: vec![ast::Statement::Let {
                        lhs: "x".into(),
                        rhs: ast::Expression::Literal(Constant::Text("Hi, mom".into()))
                    }]
                }
            })
        );
    }

    #[test]
    fn return_statement() {
        let input = &[
            T::Keyword(Keyword::Return),
            T::Literal(Literal::Integer(42)),
            T::Separator(Separator::Semicolon),
        ];
        let was = super::statement().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
            Some(Statement::Return(Expression::Literal(Constant::Int(42))))
        );

        let input = &[
            T::Keyword(Keyword::Return),
            T::Literal(Literal::Integer(1)),
            T::Separator(Separator::Plus),
            T::Literal(Literal::Integer(2)),
            T::Separator(Separator::Plus),
            T::Literal(Literal::Integer(3)),
            T::Separator(Separator::Minus),
            T::Literal(Literal::Integer(4)),
            T::Separator(Separator::Star),
            T::Literal(Literal::Integer(5)),
            T::Separator(Separator::Semicolon),
        ];
        let was = super::statement().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
            Some(Statement::Return(Expression::ApplyInfix {
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
            }))
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
        let was = super::expression().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
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
        // foo(1 + x) * 3
        let input = &[
            T::Identifier(Identifier("foo".into())),
            T::Separator(Separator::LeftParen),
            T::Literal(Literal::Integer(1)),
            T::Separator(Separator::Plus),
            T::Identifier(Identifier("x".into())),
            T::Separator(Separator::RightParen),
            T::Separator(Separator::Star),
            T::Literal(Literal::Integer(3)),
        ];
        let was = super::expression().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
            Some(Expression::ApplyInfix {
                lhs: Box::new(Expression::Apply {
                    symbol: Select::Function(Name::simple("foo")),
                    arguments: vec![Expression::ApplyInfix {
                        lhs: Box::new(Expression::Literal(Constant::Int(1))),
                        symbol: Operator::Plus,
                        rhs: Box::new(Expression::Variable(Name::simple("x")))
                    }]
                }),
                symbol: Operator::Times,
                rhs: Box::new(Expression::Literal(Constant::Int(3)))
            })
        );
    }

    #[test]
    fn function_declaration() {
        let input = &[
            T::Keyword(Keyword::Fn),
            T::Identifier(Identifier("make_hay".into())),
            T::Separator(Separator::LeftParen),
            T::Identifier(Identifier("name".into())),
            T::Separator(Separator::Colon),
            T::Identifier(Identifier("Int".into())),
            T::Separator(Separator::Comma),
            T::Identifier(Identifier("x".into())),
            T::Separator(Separator::Colon),
            T::Identifier(Identifier("Boolean".into())),
            T::Separator(Separator::RightParen),
            T::Separator(Separator::ThinRightArrow),
            T::Identifier(Identifier("Text".into())),
            T::Separator(Separator::LeftBrace),
            T::Keyword(Keyword::While),
            T::Literal(Boolean(true)),
            T::Separator(Separator::LeftBrace),
            T::Keyword(Keyword::Let),
            T::Identifier(Identifier("x".into())),
            T::Separator(Assign),
            T::Literal(FloatingPoint(427.427)),
            T::Separator(Semicolon),
            T::Separator(Separator::RightBrace),
            T::Separator(Separator::RightBrace),
        ];
        let was = super::function_declaration().parse(ParseState::new(input));
        assert_eq!(
            was.into_option(),
            Some(Declaration::Function(FunctionDef {
                name: Name::simple("make_hay"),
                parameters: vec![
                    Parameter {
                        name: "name".into(),
                        type_: Type::named(&Name::intrinsic("Int"))
                    },
                    Parameter {
                        name: "x".into(),
                        type_: Type::named(&Name::intrinsic("Boolean"))
                    },
                ],
                return_type: Type::named(&Name::intrinsic("Text")),
                body: Block {
                    statements: vec![Statement::While {
                        predicate: Expression::Literal(Constant::Boolean(true)),
                        body: Block {
                            statements: vec![Statement::Let {
                                lhs: "x".into(),
                                rhs: Expression::Literal(Constant::Float(427.427))
                            },]
                        }
                    },]
                }
            }))
        )
    }
}
