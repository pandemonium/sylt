use sylt::qombineur::{self, char, enclosed_within, separated_by, Parsed, Parsimonious, string, such_that, digit};

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Scalar(ScalarValue),
    Compound(CompoundValue),
    Null,
}

impl Value {
    pub fn parse(json_text: &str) -> Option<Self> {
        compound()
            .parse(&json_text.chars().collect::<Vec<_>>())
            .emits()
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ScalarValue {
    Number(f64),
    Text(String),
    Boolean(bool),
}

#[derive(Clone, Debug, PartialEq)]
enum CompoundValue {
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

fn ws() -> impl Parsimonious<(), Token = char> {
    such_that(|c| char::is_whitespace(*c))
        .zero_or_more()
        .ignore()
}

fn null() -> impl Parsimonious<Value, Token = char> {
    ws().skip_left(string("null").map(|_| Value::Null))
}

fn number() -> impl Parsimonious<ScalarValue, Token = char> {
    ws().skip_left(
        digit()
            .or_else(char('.'))
            .or_else(char('-'))
            .one_or_more()
            .map(|xs| xs.into_iter().collect::<String>())
            .map(|s| s.parse::<f64>().unwrap())
            .map(ScalarValue::Number),
    )
}

fn text() -> impl Parsimonious<String, Token = char> {
    ws().skip_left(
        enclosed_within(
            char('"'),
            char('"'),
            such_that(|c| c != &'"').zero_or_more(),
        )
        .map(|xs| xs.into_iter().collect::<String>()),
    )
}

fn boolean() -> impl Parsimonious<ScalarValue, Token = char> {
    ws().skip_left(
        string("true")
            .map(|_| true)
            .or_else(string("false").map(|_| false))
            .map(ScalarValue::Boolean),
    )
}

fn scalar() -> impl Parsimonious<Value, Token = char> {
    number()
        .or_else(text().map(ScalarValue::Text))
        .or_else(boolean())
        .map(Value::Scalar)
}

fn array() -> impl Parsimonious<CompoundValue, Token = char> {
    ws().skip_left(enclosed_within(
        char('['),
        char(']'),
        separated_by(value(), ws().skip_left(char(',')).ignore()),
    ))
    .map(CompoundValue::Array)
}

fn object() -> impl Parsimonious<CompoundValue, Token = char> {
    let field = text()
        .skip_right(ws().skip_left(char(':')))
        .and_also(value());

    ws().skip_left(enclosed_within(
        char('{'),
        ws().skip_left(char('}')),
        separated_by(field, ws().skip_left(char(',')).ignore()),
    ))
    .map(CompoundValue::Object)
}

fn compound() -> impl Parsimonious<Value, Token = char> {
    array().or_else(object()).map(Value::Compound)
}

fn value() -> ParsimoniousValue {
    ParsimoniousValue
}

#[derive(Clone)]
struct ParsimoniousValue;

impl Parsimonious<Value> for ParsimoniousValue {
    type Token = char;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Value, Self::Token> {
        let value = scalar().or_else(compound()).or_else(null());
        value.parse(input)
    }
}

fn main() {
    println!("Hi, world!");
}

#[cfg(test)]
mod json_tests {
    use super::{char, digit, enclosed_within, separated_by, string, such_that, Parsimonious};

    #[test]
    fn json_number() {
        let input = &char_slice(" -427.17");
        let was = value().parse(input);
        let expected = Value::Scalar(ScalarValue::Number(-427.17));
        assert_eq!(was.clone().emits(), Some(expected.clone()));
        assert_eq!(
            was.into_option(),
            Some((expected, char_slice("").as_slice()))
        )
    }

    #[test]
    fn json_text() {
        let input = &char_slice(r#" "Hi, mom""#);
        let was = value().parse(input);
        let expected = Value::Scalar(ScalarValue::Text("Hi, mom".into()));
        assert_eq!(was.clone().emits(), Some(expected.clone()));
        assert_eq!(
            was.into_option(),
            Some((expected, char_slice("").as_slice()))
        );

        let input = &char_slice(r#" """#);
        let was = value().parse(input);
        let expected = Value::Scalar(ScalarValue::Text("".into()));
        assert_eq!(was.clone().emits(), Some(expected.clone()));
        assert_eq!(
            was.into_option(),
            Some((expected, char_slice("").as_slice()))
        )
    }

    #[test]
    fn json_array() {
        let input = &char_slice(r#"[427  , "Hi, mom"]"#);
        let was = value().parse(input);
        let expected = Value::Compound(CompoundValue::Array(vec![
            Value::Scalar(ScalarValue::Number(427.0)),
            Value::Scalar(ScalarValue::Text("Hi, mom".into())),
        ]));
        assert_eq!(was.clone().emits(), Some(expected.clone()));
        assert_eq!(
            was.into_option(),
            Some((expected, char_slice("").as_slice()))
        );
    }

    #[test]
    fn json_object() {
        let input = &char_slice(
            r#" { 
            "quux" : 427 ,   
            "hi,"  : " mom"  
            , "moms":[ 428  ,"mothers"],"yes": null
            ,"no"    :true,
            "empty_object" : { } ,"empty_array":[]
        }"#,
        );
        let was = value().parse(input);
        let expected = Value::Compound(CompoundValue::Object(vec![
            ("quux".into(), Value::Scalar(ScalarValue::Number(427.0))),
            (
                "hi,".into(),
                Value::Scalar(ScalarValue::Text(" mom".into())),
            ),
            (
                "moms".into(),
                Value::Compound(CompoundValue::Array(vec![
                    Value::Scalar(ScalarValue::Number(428.0)),
                    Value::Scalar(ScalarValue::Text("mothers".into())),
                ])),
            ),
            ("yes".into(), Value::Null),
            ("no".into(), Value::Scalar(ScalarValue::Boolean(true))),
            (
                "empty_object".into(),
                Value::Compound(CompoundValue::Object(vec![])),
            ),
            (
                "empty_array".into(),
                Value::Compound(CompoundValue::Array(vec![])),
            ),
        ]));
        assert_eq!(was.clone().emits(), Some(expected.clone()));
        assert_eq!(
            was.into_option(),
            Some((expected, char_slice("").as_slice()))
        );
    }
}
