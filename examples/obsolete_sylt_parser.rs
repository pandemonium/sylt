use sylt::ast;
use combine::parser::char::{char, digit, letter, spaces, string};
use combine::parser::combinator::recognize;
use combine::stream::ResetStream;
use combine::{
    attempt, between, choice, many, many1, none_of, not_followed_by, optional, parser,
    sep_by, sep_by1, skip_many1, EasyParser, ParseError, Parser, Positioned,
    StreamOnce,
};
use std::num::{ParseFloatError, ParseIntError};

fn semi<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char(';'))
}

fn assign<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char('='))
}

fn left_brace<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char('{'))
}

fn right_brace<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char('}'))
}

fn left_paren<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char('('))
}

fn right_paren<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char(')'))
}

fn comma<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char(','))
}

fn star<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char('*'))
}

fn plus<'a, I>() -> impl Parser<I, Output = char>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(char('+'))
}

pub fn keyword<'a, I>() -> impl Parser<I, Output = &'a str>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(choice!(
        string("let"),
        string("while"),
        string("else"),
        string("if"),
        string("fn"),
        string("return")
    ))
}

pub fn identifier<'a, I>() -> impl Parser<I, Output = String>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    spaces().with(not_followed_by(keyword()).with(many(letter().or(char('_')))))
}

parser! {
    #[inline]
    pub fn factor[I]()(I) -> ast::Expression
    where [
        I: StreamOnce<Token = char> + Positioned + ResetStream,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
    ] {
        expression_rhs() //choice!(, left_paren().with(expression().skip(right_paren())))
    }
}

parser! {
    #[inline]
    pub fn term[I]()(I) -> ast::Expression
    where [
        I: StreamOnce<Token = char> + Positioned + ResetStream,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
    ] {
        spaces().with(
            factor()
//            sep_by1(factor(), star()).map(|xs: Vec<_>| {
//                xs.into_iter().reduce(|e0, e1| ast::Expression::ApplyInfix {
//                    lhs: Box::new(e0),
//                    symbol: ast::Operator::Times,
//                    rhs: Box::new(e1),
//                }).unwrap() // sep_by1 promises at least once.
//            })
        )
//        spaces().with(
//            factor().and(optional(sep_by(factor(), star()))).map(|(x, xs): (ast::Expression, Option<Vec<_>>)| {
//                if let Some(mut xs) = xs {
//                    xs.push(x);
//                    xs.into_iter().reduce(|e0, e1| ast::Expression::ApplyInfix {
//                        lhs: Box::new(e0),
//                        symbol: ast::Operator::Times,
//                        rhs: Box::new(e1),
//                    }).unwrap()
//                } else {
//                    x
//                }
//            })
//        )

    }
}

parser! {
    #[inline]
    pub fn expression[I]()(I) -> ast::Expression
    where [
        I: StreamOnce<Token = char> + Positioned + ResetStream,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
    ]
    {
        spaces().with(
            term() //.and(optional((plus(), term()))).map(|(x, _y)| x)
        )

//        spaces().with(
//            sep_by1(term(), plus()).map(|xs: Vec<_>| {
//                xs.into_iter().reduce(|e0, e1| ast::Expression::ApplyInfix {
//                    lhs: Box::new(e0),
//                    symbol: ast::Operator::Plus,
//                    rhs: Box::new(e1),
//                }).unwrap() // sep_by1 promises at least once.
//            })
//        )

//        spaces().with(
//            term().and(optional(sep_by(term(), plus()))).map(|(x, xs): (ast::Expression, Option<Vec<_>>)| {
//                if let Some(mut xs) = xs {
//                    xs.push(x);
//                    xs.into_iter().reduce(|e0, e1| ast::Expression::ApplyInfix {
//                        lhs: Box::new(e0),
//                        symbol: ast::Operator::Plus,
//                        rhs: Box::new(e1),
//                    }).unwrap()
//                } else {
//                    x
//                }
//            })
//        )
    }
}

parser! {
    #[inline]
    pub fn expression_rhs[I]()(I) -> ast::Expression
    where [
        I: StreamOnce<Token = char> + Positioned + ResetStream,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
    ]
    {
        let var_ref = spaces().with(identifier()).map(|x| ast::Expression::Variable(ast::Name::simple(&x)));
        choice!(
            literal_constant(),
            apply_function(),
            var_ref
        )
    }
}

parser! {
    #[inline]
    pub fn statement[I]()(I) -> ast::Statement
    where [
        I: StreamOnce<Token = char> + Positioned + ResetStream,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
    ]
    {
        // foo();
        let expression_statement = attempt(apply_function())
            .skip(semi())
            .map(ast::Statement::Expression);

        choice!(
            attempt(let_statement()).expected("let statement"),
            attempt(while_statement()).expected("while statement"),
            attempt(if_statement()).expected("if statement"),
            attempt(return_statement()).expected("return statement"),
            attempt(expression_statement).expected("expression statement")
        )
    }
}

pub fn if_statement<'a, I>() -> impl Parser<I, Output = ast::Statement>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
{
    let if_then = spaces().with((string("if").with(expression()), block()));
    let if_else = spaces().with(string("else").with(block()));

    (if_then, if_else).map(|((predicate, when_true), when_false)| ast::Statement::If {
        predicate,
        when_true,
        when_false,
    })
}

pub fn while_statement<'a, I>() -> impl Parser<I, Output = ast::Statement>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
{
    spaces().with(
        string("while")
            .with((expression(), block()))
            .map(|(predicate, body)| ast::Statement::While { predicate, body }),
    )
}

pub fn block<'a, I>() -> impl Parser<I, Output = ast::Block>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
{
    spaces().with(
        between(left_brace(), right_brace(), many(statement()))
            .map(|statements| ast::Block { statements }),
    )
}

pub fn let_statement<'a, I>() -> impl Parser<I, Output = ast::Statement>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
{
    spaces()
        .with((
            string("let").with(identifier()).skip(assign()),
            expression().skip(semi()),
        ))
        .map(|(lhs, rhs)| ast::Statement::Let { lhs, rhs })
}

pub fn return_statement<'a, I>() -> impl Parser<I, Output = ast::Statement>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
{
    spaces()
        .with(string("return").with(expression().skip(semi())))
        .map(ast::Statement::Return)
}

pub fn literal_constant<'a, I>() -> impl Parser<I, Output = ast::Expression>
where
    I: StreamOnce<Token = char> + Positioned + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
    <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
{
    let boolean = string("true")
        .map(|_| true)
        .or(string("false").map(|_| false))
        .map(ast::Constant::Boolean);

    let float = recognize((skip_many1(digit()), char('.'), skip_many1(digit())))
        .and_then(|x: String| x.parse::<f64>())
        .map(ast::Constant::Float);

    let int = many1(digit())
        .and_then(|x: String| x.parse::<i64>())
        .map(ast::Constant::Int);

    let text = between(char('"'), char('"'), many(none_of(vec!['"']))).map(ast::Constant::Text);

    let unit = string("()").map(|_| ast::Constant::Void);

    spaces()
        .with(choice!(boolean, attempt(float), int, text, unit))
        .map(ast::Expression::Literal)
}

parser! {
    pub fn apply_function[I]()(I) -> ast::Expression
    where
    [
        I: StreamOnce<Token = char> + Positioned + ResetStream,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseFloatError>,
        <I::Error as ParseError<I::Token, I::Range, I::Position>>::StreamError: From<ParseIntError>,
    ]
    {
        let expression_list = sep_by(expression(), comma())
            .map(|xs: Vec<ast::Expression>| xs);
        spaces()
            .with((
                attempt(identifier()).map(|x| ast::Select::Function(ast::Name::simple(&x))),
                between(left_paren(), right_paren(), expression_list),
            ))
            .map(|(symbol, arguments)| ast::Expression::Apply { symbol, arguments })
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::{ast, parser};
    use combine::EasyParser;

    #[test]
    fn let_statement() {
        let source = r#"let foo="Hi, mom";"#;
        assert_eq!(
            ast::Statement::Let {
                lhs: "foo".into(),
                rhs: ast::Expression::Literal(ast::Constant::Text("Hi, mom".into()))
            },
            parser::statement().easy_parse(source).unwrap().0
        );
    }

    #[test]
    fn apply_function() {
        {
            let source = r#"  println ( 42.01,"Hi, mom")   "#;
            assert_eq!(
                ast::Expression::Apply {
                    symbol: ast::Select::Function(ast::Name::simple("println")),
                    arguments: vec![
                        ast::Expression::Literal(ast::Constant::Float(42.01)),
                        ast::Expression::Literal(ast::Constant::Text("Hi, mom".into())),
                    ]
                },
                parser::apply_function().easy_parse(source).unwrap().0
            )
        }
        {
            let source = r#"println()"#;
            assert_eq!(
                ast::Expression::Apply {
                    symbol: ast::Select::Function(ast::Name::simple("println")),
                    arguments: vec![]
                },
                parser::apply_function().easy_parse(source).unwrap().0
            )
        }
        {
            let source = r#"  println ( false,  42.01,"Hi, mom")   "#;
            assert_eq!(
                ast::Expression::Apply {
                    symbol: ast::Select::Function(ast::Name::simple("println")),
                    arguments: vec![
                        ast::Expression::Literal(ast::Constant::Boolean(false)),
                        ast::Expression::Literal(ast::Constant::Float(42.01)),
                        ast::Expression::Literal(ast::Constant::Text("Hi, mom".into())),
                    ]
                },
                parser::apply_function().easy_parse(source).unwrap().0
            )
        }
        {
            let source = r#"  println ( false,  your_mum("hi"))   "#;
            assert_eq!(
                ast::Expression::Apply {
                    symbol: ast::Select::Function(ast::Name::simple("println")),
                    arguments: vec![
                        ast::Expression::Literal(ast::Constant::Boolean(false)),
                        ast::Expression::Apply {
                            symbol: ast::Select::Function(ast::Name::simple("your_mum")),
                            arguments: vec![ast::Expression::Literal(ast::Constant::Text(
                                "hi".into()
                            )),]
                        },
                    ]
                },
                parser::apply_function().easy_parse(source).unwrap().0
            )
        }
    }

    #[test]
    fn statement() {
        {
            let source = r#"  println ( false,  42.01,"Hi, mom"); "#;
            assert_eq!(
                ast::Statement::Expression(ast::Expression::Apply {
                    symbol: ast::Select::Function(ast::Name::simple("println")),
                    arguments: vec![
                        ast::Expression::Literal(ast::Constant::Boolean(false)),
                        ast::Expression::Literal(ast::Constant::Float(42.01)),
                        ast::Expression::Literal(ast::Constant::Text("Hi, mom".into())),
                    ]
                }),
                parser::statement().easy_parse(source).unwrap().0
            )
        }
        {
            let source = r#"let quux = println ( false, 42.01, "Hi, mom"); "#;
            assert_eq!(
                ast::Statement::Let {
                    lhs: "quux".into(),
                    rhs: ast::Expression::Apply {
                        symbol: ast::Select::Function(ast::Name::simple("println")),
                        arguments: vec![
                            ast::Expression::Literal(ast::Constant::Boolean(false)),
                            ast::Expression::Literal(ast::Constant::Float(42.01)),
                            ast::Expression::Literal(ast::Constant::Text("Hi, mom".into())),
                        ]
                    }
                },
                parser::statement().easy_parse(source).unwrap().0
            )
        }
        {
            let source = r#" if your_mum ("hi"){println(427.5);} else {exit(); } "#;
            assert_eq!(
                ast::Statement::If {
                    predicate: ast::Expression::Apply {
                        symbol: ast::Select::Function(ast::Name::simple("your_mum")),
                        arguments: vec![ast::Expression::Literal(ast::Constant::Text("hi".into()))]
                    },
                    when_true: ast::Block {
                        statements: vec![ast::Statement::Expression(ast::Expression::Apply {
                            symbol: ast::Select::Function(ast::Name::simple("println")),
                            arguments: vec![ast::Expression::Literal(ast::Constant::Float(427.5)),]
                        })]
                    },
                    when_false: ast::Block {
                        statements: vec![ast::Statement::Expression(ast::Expression::Apply {
                            symbol: ast::Select::Function(ast::Name::simple("exit")),
                            arguments: vec![]
                        })]
                    }
                },
                parser::statement().easy_parse(source).unwrap().0
            )
        }
        {
            let source =
                r#"while your_mum(9001){let merry_goaround="hi; mum";save(merry_goaround);}"#;
            assert_eq!(
                ast::Statement::While {
                    predicate: ast::Expression::Apply {
                        symbol: ast::Select::Function(ast::Name::simple("your_mum")),
                        arguments: vec![ast::Expression::Literal(ast::Constant::Int(9001))]
                    },
                    body: ast::Block {
                        statements: vec![
                            ast::Statement::Let {
                                lhs: "merry_goaround".into(),
                                rhs: ast::Expression::Literal(ast::Constant::Text(
                                    "hi; mum".into()
                                ))
                            },
                            ast::Statement::Expression(ast::Expression::Apply {
                                symbol: ast::Select::Function(ast::Name::simple("save")),
                                arguments: vec![ast::Expression::Variable(ast::Name::simple(
                                    "merry_goaround"
                                ))]
                            })
                        ]
                    }
                },
                parser::statement().easy_parse(source).unwrap().0
            )
        }
        {
            let source = r#"return din_mamma;"#;
            assert_eq!(
                ast::Statement::Return(ast::Expression::Variable(ast::Name::simple(&"din_mamma"))),
                parser::statement().easy_parse(source).unwrap().0
            )
        }
    }
}
