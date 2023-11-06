use std::{fmt, marker};

#[derive(Clone, PartialEq, Debug)]
pub enum Parsed<'a, A, Token> {
    Emits(A, &'a [Token]),
    Diverges,
}

impl<'a, A, Token> Parsed<'a, A, Token> {
    pub fn map<F, B>(self, f: F) -> Parsed<'a, B, Token>
    where
        F: FnOnce(A) -> B,
    {
        match self {
            Self::Emits(x, remains) => Parsed::Emits(f(x), remains),
            Self::Diverges => Parsed::Diverges,
        }
    }

    pub fn flat_map<F, B>(self, f: F) -> Parsed<'a, B, Token>
    where
        F: FnOnce(A, &'a [Token]) -> Parsed<'a, B, Token>,
    {
        match self {
            Parsed::Emits(x, remains) => f(x, remains),
            Parsed::Diverges => Parsed::Diverges,
        }
    }

    pub fn filter_map<F, B>(self, p: F) -> Parsed<'a, B, Token>
    where
        F: FnOnce(&A) -> Option<B>,
    {
        match self.map(|x| p(&x)) {
            Parsed::Emits(Some(b), remains) => Parsed::Emits(b, remains),
            _otherwise => Parsed::Diverges,
        }
    }

    pub fn emits(self) -> Option<A> {
        if let Self::Emits(x, _) = self {
            Some(x)
        } else {
            None
        }
    }

    pub fn into_option(self) -> Option<(A, &'a [Token])> {
        if let Self::Emits(x, remains) = self {
            Some((x, remains))
        } else {
            None
        }
    }
}

pub trait Parsimonious<A>: Clone + Sized {
    // move Clone and Sized to impl
    type Token: Clone;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token>;

    fn parse_phrase<'a>(self, input: &'a [Self::Token]) -> Option<A> {
        self.parse(input)
            .into_option()
            .and_then(|(x, remains)| remains.is_empty().then_some(x))
    }

    fn map<F, B>(self, f: F) -> Map<F, A, Self>
    where
        F: FnOnce(A) -> B,
    {
        Map {
            inner: self,
            f,
            _emits: marker::PhantomData,
        }
    }

    fn flat_map<F, B, Q>(self, f: F) -> FlatMap<F, A, Self>
    where
        Q: Parsimonious<B>,
        F: FnOnce(A) -> Q,
    {
        FlatMap {
            inner: self,
            f,
            _emits: marker::PhantomData,
        }
    }

    fn filter_map<F, B>(self, p: F) -> FilterMap<F, A, B, Self>
    where
        F: FnOnce(&A) -> Option<B>,
    {
        FilterMap {
            inner: self,
            p,
            _maps_to: marker::PhantomData,
            _emits: marker::PhantomData,
        }
    }

    fn or_else<Q>(self, rhs: Q) -> OrElse<Self, Q>
    where
        Q: Parsimonious<A, Token = Self::Token>,
    {
        OrElse { lhs: self, rhs }
    }

    fn and_also<B, Q>(self, rhs: Q) -> AndAlso<Self, Q>
    where
        Q: Parsimonious<B, Token = Self::Token>,
    {
        AndAlso { lhs: self, rhs }
    }

    fn optionally(self) -> Optionally<Self> {
        Optionally(self)
    }

    fn skip_right<Q, Z>(self, consequent: Q) -> SkipConsequent<Self, Q, Z> {
        skip_consequent(self, consequent)
    }

    fn skip_left<Q, Z>(self, antecedent: Q) -> SkipAntecedent<Self, Q, Z> {
        skip_antecedent(self, antecedent)
    }

    fn ignore<B>(self) -> Ignore<Self, B> {
        Ignore(self, marker::PhantomData)
    }

    fn zero_or_more(self) -> ZeroOrMore<Self> {
        ZeroOrMore(self)
    }

    fn one_or_more(self) -> OneOrMore<Self> {
        OneOrMore(self)
    }

    fn with_name(self) -> WithName<Self, A> {
        WithName(self, marker::PhantomData)
    }
}

// How would this thing work?
// A fmt::Display that matches parsers wrapping others and can
// pretty-print a tree?
#[derive(Clone, Copy)]
pub struct WithName<P, A>(P, marker::PhantomData<A>);

impl<P, A> Parsimonious<A> for WithName<P, A>
where
    P: Parsimonious<A>,
    P::Token: Clone,
    A: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token> {
        let WithName(inner, ..) = self;
        inner.parse(input)
    }
}

#[derive(Clone, Copy)]
pub struct Ignore<P, A>(P, marker::PhantomData<A>);

impl<P, A> Parsimonious<()> for Ignore<P, A>
where
    P: Parsimonious<A>,
    P::Token: Clone,
    A: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, (), Self::Token> {
        let Ignore(inner, ..) = self;
        inner.map(|_| ()).parse(input)
    }
}

pub fn skip_antecedent<P, Q, C>(antecedent: P, consequent: Q) -> SkipAntecedent<P, Q, C> {
    SkipAntecedent(antecedent, consequent, marker::PhantomData)
}

#[derive(Clone, Copy)]
pub struct SkipAntecedent<P, Q, C>(P, Q, marker::PhantomData<C>);

impl<P, Q, A, C> Parsimonious<C> for SkipAntecedent<P, Q, A>
where
    P: Parsimonious<A>,
    Q: Parsimonious<C, Token = P::Token>,
    P::Token: Clone,
    A: Clone,
    C: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, C, Self::Token> {
        let SkipAntecedent(antecedent, consequent, ..) = self;
        antecedent.and_also(consequent).map(|(_, a)| a).parse(input)
    }
}

pub fn skip_consequent<P, Q, C>(antecedent: P, consequent: Q) -> SkipConsequent<P, Q, C> {
    SkipConsequent(antecedent, consequent, marker::PhantomData)
}

impl<P, Q, A, C> Parsimonious<A> for SkipConsequent<P, Q, C>
where
    P: Parsimonious<A>,
    Q: Parsimonious<C, Token = P::Token>,
    P::Token: Clone,
    A: Clone,
    C: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token> {
        let SkipConsequent(antecedent, consequent, ..) = self;
        antecedent.and_also(consequent).map(|(a, _)| a).parse(input)
    }
}

#[derive(Clone, Copy)]
pub struct SkipConsequent<P, Q, C>(P, Q, marker::PhantomData<C>);

pub fn diverge<T>() -> Diverged<T> {
    Diverged(marker::PhantomData)
}

#[derive(Clone, Copy)]
pub struct Diverged<T>(marker::PhantomData<T>);

impl<T> Parsimonious<()> for Diverged<T>
where
    T: Clone,
{
    type Token = T;

    fn parse<'a>(self, _input: &'a [Self::Token]) -> Parsed<'a, (), Self::Token> {
        Parsed::Diverges
    }
}

#[derive(Clone, Copy)]
pub struct OneOrMore<P>(P);

// Re-write in terms of Extends trait
impl<P, A> Parsimonious<Vec<A>> for OneOrMore<P>
where
    P: Parsimonious<A>,
    P::Token: Clone,
    A: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<A>, Self::Token> {
        let OneOrMore(inner) = self;

        inner
            .clone()
            .and_also(inner.zero_or_more())
            .map(|(x, mut xs)| {
                xs.insert(0, x);
                xs
            })
            .parse(input)
    }
}

// This could be re-written such that Option is not the default
#[derive(Clone, Copy)]
pub struct Optionally<P>(P);

impl<P, A> Parsimonious<Option<A>> for Optionally<P>
where
    P: Parsimonious<A>,
    P::Token: Clone,
    A: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Option<A>, Self::Token> {
        let Optionally(inner) = self;
        inner.map(Some).or_else(empty(None)).parse(input)
    }
}

#[derive(Clone, Copy)]
pub struct ZeroOrMore<P>(P);

impl<P, A> Parsimonious<Vec<A>> for ZeroOrMore<P>
where
    P: Parsimonious<A>,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<A>, Self::Token> {
        let ZeroOrMore(inner) = &self;
        let mut cursor = input;
        let mut result: Vec<A> = vec![];

        loop {
            match inner.clone().parse(cursor) {
                Parsed::Emits(x, remains) => {
                    result.push(x);
                    cursor = remains;
                }
                Parsed::Diverges => break,
            }
        }

        Parsed::Emits(result, cursor)
    }
}

#[derive(Clone, Copy)]
pub struct FilterMap<F, A, B, P> {
    inner: P,
    p: F,
    _maps_to: marker::PhantomData<A>,
    _emits: marker::PhantomData<B>,
}

impl<F, A, B, P> Parsimonious<B> for FilterMap<F, A, B, P>
where
    P: Parsimonious<A>,
    P::Token: Clone,
    F: FnOnce(&A) -> Option<B> + Clone,
    A: Clone,
    B: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, B, Self::Token> {
        let FilterMap { inner, p, .. } = self;
        inner.parse(input).filter_map(p)
    }
}

#[derive(Clone, Copy)]
pub struct Map<F, A, P> {
    inner: P,
    f: F,
    _emits: marker::PhantomData<A>,
}

impl<F, A, B, P> Parsimonious<B> for Map<F, A, P>
where
    P: Parsimonious<A>,
    P::Token: Clone,
    F: FnOnce(A) -> B + Clone,
    A: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, B, Self::Token> {
        self.inner.parse(input).map(self.f)
    }
}

#[derive(Clone, Copy)]
pub struct FlatMap<F, A, P> {
    inner: P,
    f: F,
    _emits: marker::PhantomData<A>,
}

impl<F, A, B, P, Q> Parsimonious<B> for FlatMap<F, A, P>
where
    P: Parsimonious<A>,
    Q: Parsimonious<B, Token = P::Token>,
    F: FnOnce(A) -> Q + Clone,
    A: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, B, Self::Token> {
        self.inner
            .parse(input)
            .flat_map(|x, remains| (self.f)(x).parse(remains))
    }
}

pub fn empty<A, T>(produces: A) -> Empty<A, T>
where
    A: Clone,
{
    Empty {
        produces,
        _token_type: marker::PhantomData,
    }
}

#[derive(Clone, Copy)]
pub struct Empty<A, T> {
    produces: A,
    _token_type: marker::PhantomData<T>,
}

impl<A, T> Parsimonious<A> for Empty<A, T>
where
    A: Clone,
    T: Clone,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token> {
        Parsed::Emits(self.produces.clone(), input)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Any<T>(marker::PhantomData<T>);

pub fn any<T>() -> Any<T> {
    Any(marker::PhantomData)
}

impl<T> Parsimonious<T> for Any<T>
where
    T: Clone,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, T, Self::Token> {
        match input {
            [head, remains @ ..] => Parsed::Emits(head.clone(), remains),
            _otherwise => Parsed::Diverges,
        }
    }
}

pub fn such_that<T, F>(predicate: F) -> SuchThat<F, T>
where
    F: Fn(&T) -> bool,
{
    SuchThat {
        predicate,
        _token_type: marker::PhantomData,
    }
}

#[derive(Clone, Copy)]
pub struct SuchThat<F, T> {
    predicate: F,
    _token_type: marker::PhantomData<T>,
}

impl<F, T> Parsimonious<T> for SuchThat<F, T>
where
    F: Fn(&T) -> bool + Clone,
    T: Clone,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, T, Self::Token> {
        match input {
            [head, remains @ ..] if (self.predicate)(head) => Parsed::Emits(head.clone(), remains),
            _otherwise => Parsed::Diverges,
        }
    }
}

pub fn this<T>(token: T) -> impl Parsimonious<T, Token = T>
where
    T: PartialEq + Clone,
{
    such_that(move |x| x == &token)
}

pub fn char(c: char) -> impl Parsimonious<char, Token = char> {
    this(c)
}

pub fn letter() -> impl Parsimonious<char, Token = char> {
    such_that(|c| char::is_alphabetic(*c))
}

pub fn digit() -> impl Parsimonious<char, Token = char> {
    such_that(|c| char::is_digit(*c, 10))
}

pub fn one_of<T>(choice: &[T]) -> impl Parsimonious<T, Token = T> + Clone
where
    T: PartialEq + Clone,
{
    let choice = choice.to_owned();
    such_that(move |c| choice.contains(c))
}

pub fn take<T>(i: usize) -> Take<T> {
    Take(i, marker::PhantomData)
}

#[derive(Clone, Copy)]
pub struct Take<T>(usize, marker::PhantomData<T>);

impl<T> Parsimonious<Vec<T>> for Take<T>
where
    T: Clone,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<T>, Self::Token> {
        let Take(length, ..) = self;
        if input.len() >= length {
            Parsed::Emits(
                input[0..length].into_iter().cloned().collect::<Vec<_>>(),
                &input[length..],
            )
        } else {
            Parsed::Diverges
        }
    }
}

pub fn string(literal: &str) -> LiteralMatch<char> {
    LiteralMatch(literal.chars().collect())
}

pub fn literal<T>(literal: &[T]) -> LiteralMatch<T>
where
    T: Clone,
{
    LiteralMatch(literal.to_vec())
}

#[derive(Clone)]
pub struct LiteralMatch<T>(Vec<T>);

impl<T> Parsimonious<Vec<T>> for LiteralMatch<T>
where
    T: Clone + PartialEq,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<T>, Self::Token> {
        let LiteralMatch(literal) = self;
        let literal_length = literal.len();

        if input.len() >= literal_length && literal.into_iter().zip(input).all(|(p, q)| &p == q) {
            Parsed::Emits(input[0..literal_length].to_vec(), &input[literal_length..])
        } else {
            Parsed::Diverges
        }
    }
}

pub fn separated_by<W, S, T>(word: W, separator: S) -> SeparatedBy<W, S, T> {
    SeparatedBy {
        word,
        separator,
        _token_type: marker::PhantomData,
    }
}

#[derive(Clone, Copy)]
pub struct SeparatedBy<W, S, T> {
    word: W,
    separator: S,
    _token_type: marker::PhantomData<T>,
}

impl<A, W, S, T> Parsimonious<Vec<A>> for SeparatedBy<W, S, T>
where
    W: Parsimonious<A, Token = T>,
    S: Parsimonious<(), Token = W::Token>,
    W::Token: Clone,
    A: Clone,
    T: Clone,
{
    type Token = W::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<A>, Self::Token> {
        let SeparatedBy {
            word, separator, ..
        } = self;

        word.clone()
            .and_also(
                separator
                    .and_also(word)
                    .map(|(_, word)| word)
                    .zero_or_more(),
            )
            .map(|(head, mut tail)| {
                tail.insert(0, head);
                tail
            })
            .or_else(empty(vec![])) // separated_by1 would not have this clause
            .parse(input)
    }
}

pub fn enclosed_within<O, C, P, A, B, T>(
    open: O,
    close: C,
    body: P,
) -> EnclosedWithin<O, C, P, A, B, T> {
    EnclosedWithin {
        open,
        close,
        body,
        _o_type: marker::PhantomData,
        _c_type: marker::PhantomData,
        _token_type: marker::PhantomData,
    }
}

#[derive(Clone, Copy)]
pub struct EnclosedWithin<O, C, P, A, B, T> {
    open: O,
    close: C,
    body: P,
    _o_type: marker::PhantomData<A>,
    _c_type: marker::PhantomData<B>,
    _token_type: marker::PhantomData<T>,
}

impl<O, C, P, A, X, Y, T> Parsimonious<A> for EnclosedWithin<O, C, P, X, Y, T>
where
    P: Parsimonious<A, Token = T>,
    O: Parsimonious<X, Token = P::Token>,
    C: Parsimonious<Y, Token = P::Token>,
    P::Token: Clone,
    A: Clone,
    X: Clone,
    Y: Clone,
    T: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token> {
        let EnclosedWithin {
            open, close, body, ..
        } = self;
        open.map(|_| ())
            .and_also(body)
            .and_also(close.map(|_| ()))
            .map(|((_, a), _)| a)
            .parse(input)
    }
}

#[derive(Clone, Copy)]
pub struct OrElse<P, Q> {
    lhs: P,
    rhs: Q,
}

impl<A, P, Q, T> Parsimonious<A> for OrElse<P, Q>
where
    P: Parsimonious<A, Token = T>,
    P::Token: Clone,
    Q: Parsimonious<A, Token = P::Token>,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token> {
        if let return_value @ Parsed::Emits(..) = self.lhs.parse(input) {
            return_value
        } else {
            self.rhs.parse(input)
        }
    }
}

#[derive(Clone, Copy)]
pub struct AndAlso<P, Q> {
    lhs: P,
    rhs: Q,
}

impl<A, B, P, Q> Parsimonious<(A, B)> for AndAlso<P, Q>
where
    P: Parsimonious<A>,
    Q: Parsimonious<B, Token = P::Token>,
    P::Token: Clone,
    A: Clone,
    B: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, (A, B), Self::Token> {
        self.lhs
            .flat_map(|lhs| self.rhs.map(|rhs| (lhs, rhs)))
            .parse(input)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        any, char, digit, diverge, empty, enclosed_within, separated_by, string, such_that, take,
        this, Parsed, Parsimonious,
    };

    fn char_slice(text: &str) -> Vec<char> {
        text.chars().collect::<Vec<_>>()
    }

    #[test]
    fn empty_parse() {
        let input = &char_slice("hi, mom");
        let was = empty(427).parse(input);
        assert_eq!(was.clone().emits(), Some(427));
        assert_eq!(was.into_option(), Some((427, input.as_slice())));
    }

    #[test]
    fn any_parse() {
        let input = &char_slice("hi, mom");
        let was = any().parse(input);
        assert_eq!(was.clone().emits(), Some('h'));
        assert_eq!(was.into_option(), Some(('h', &input.as_slice()[1..])));
    }

    #[test]
    fn or_else() {
        let input = &char_slice("hi, mom");
        let empty_input = &"".chars().collect::<Vec<_>>();

        let was = any().or_else(empty('x')).parse(input);
        assert_eq!(was.clone().emits(), Some('h'));
        assert_eq!(was.into_option(), Some(('h', &input.as_slice()[1..])));

        let was = any().or_else(empty('x')).parse(empty_input);
        assert_eq!(was.clone().emits(), Some('x'));
        assert_eq!(was.into_option(), Some(('x', empty_input.as_slice())));
    }

    #[test]
    fn and_also() {
        let input = &char_slice("hi, mom");
        let empty_input = &"".chars().collect::<Vec<_>>();

        let was = any().and_also(any()).parse(input);
        assert_eq!(was.clone().emits(), Some(('h', 'i')));
        assert_eq!(
            was.into_option(),
            Some((('h', 'i'), &input.as_slice()[2..]))
        );

        let was = any().and_also(empty('x')).parse(input);
        assert_eq!(was.clone().emits(), Some(('h', 'x')));
        assert_eq!(
            was.into_option(),
            Some((('h', 'x'), &input.as_slice()[1..]))
        );

        let was = empty('x').and_also(any()).parse(input);
        assert_eq!(was.clone().emits(), Some(('x', 'h')));
        assert_eq!(
            was.into_option(),
            Some((('x', 'h'), &input.as_slice()[1..]))
        );

        let was = any().and_also(empty('x')).parse(empty_input);
        assert_eq!(was, Parsed::Diverges);
    }

    #[test]
    fn such_that_test() {
        let input = &char_slice("hi, mom");
        let empty_input = &"".chars().collect::<Vec<_>>();

        let was = such_that(|x| x == &'h').parse(input);
        assert_eq!(was.clone().emits(), Some('h'));
        assert_eq!(was.into_option(), Some(('h', &input.as_slice()[1..])));

        let was = such_that(|x| x == &'x').parse(empty_input);
        assert_eq!(was.clone().emits(), None);
        assert_eq!(was.into_option(), None);
    }

    #[test]
    fn single_tokens() {
        let input = &char_slice("hi, mom");
        let was = char('h')
            .and_also(this('i'))
            .and_also(such_that(|x| x == &','))
            .parse(input);

        assert_eq!(was.clone().emits(), Some((('h', 'i'), ',')));
        assert_eq!(was.into_option(), Some(((('h', 'i'), ','), &input[3..])));
    }

    #[test]
    fn map() {
        let input = &char_slice("hi, mom");
        let was = char('h')
            .and_also(this('i'))
            .and_also(such_that(|x| x == &','))
            .map(|((_a, _b), _c)| 427)
            .parse(input);

        assert_eq!(was.clone().emits(), Some(427));
        assert_eq!(was.into_option(), Some((427, &input[3..])));
    }

    #[test]
    fn flat_map() {
        let input = &char_slice("hi, mom");
        let was = char('h')
            .and_also(this('i'))
            .and_also(such_that(|x| x == &','))
            .and_also(char(' '))
            .flat_map(|(((_a, _b), _c), _d)| char('m'))
            .parse(input);

        assert_eq!(was.clone().emits(), Some('m'));
        assert_eq!(was.into_option(), Some(('m', &input[5..])));
    }

    #[test]
    fn takes() {
        let input = &char_slice("hi, mom");
        let was = take(3).parse(input);

        assert_eq!(was.clone().emits(), Some(vec!['h', 'i', ',']));
        assert_eq!(was.into_option(), Some((vec!['h', 'i', ','], &input[3..])));
    }

    #[test]
    fn strings() {
        let input = &char_slice("hi, mom");
        let p = "hi, ";
        let q = "mom";

        let was = string(p).parse(input);
        assert_eq!(was.clone().emits(), Some(char_slice(p)));
        assert_eq!(was.into_option(), Some((char_slice(p), &input[4..])));

        let was = string(p).and_also(string(q)).parse(input);
        assert_eq!(was.clone().emits(), Some((char_slice(p), char_slice(q),)));
        assert_eq!(
            was.into_option(),
            Some(((char_slice(p), char_slice(q),), char_slice("").as_slice()))
        )
    }

    #[test]
    fn zero_or_more() {
        let input = &char_slice("aaaaabbbbc");
        let was = char('a').zero_or_more().parse(input);

        assert_eq!(was.emits(), Some(char_slice("aaaaa")));

        let was = char('a')
            .zero_or_more()
            .map(|x| x.into_iter().collect::<String>())
            .and_also(
                char('b')
                    .zero_or_more()
                    .map(|x| x.into_iter().collect::<String>()),
            )
            .and_also(
                char('c')
                    .zero_or_more()
                    .map(|x| x.into_iter().collect::<String>()),
            )
            .and_also(
                char('d')
                    .zero_or_more()
                    .map(|x| x.into_iter().collect::<String>()),
            )
            .parse(input);

        assert_eq!(
            was.clone().emits(),
            Some(((("aaaaa".into(), "bbbb".into()), "c".into()), "".into()))
        );

        assert_eq!(
            was.into_option(),
            Some((
                ((("aaaaa".into(), "bbbb".into()), "c".into()), "".into()),
                char_slice("").as_slice()
            ))
        )
    }

    #[test]
    fn one_or_more() {
        let input = &char_slice("aaaaabbbbc");
        let p = char('a')
            .one_or_more()
            .map(|x| x.into_iter().collect::<String>())
            .and_also(
                char('b')
                    .one_or_more()
                    .map(|x| x.into_iter().collect::<String>()),
            )
            .and_also(
                char('c')
                    .one_or_more()
                    .map(|x| x.into_iter().collect::<String>()),
            );

        let q = p.clone().and_also(
            char('d')
                .one_or_more()
                .map(|x| x.into_iter().collect::<String>()),
        );

        let was = p.parse(input);

        assert_eq!(
            was.clone().emits(),
            Some((("aaaaa".into(), "bbbb".into()), "c".into()))
        );

        assert_eq!(
            was.into_option(),
            Some((
                ((("aaaaa".into(), "bbbb".into()), "c".into())),
                char_slice("").as_slice()
            ))
        );

        let was = q.parse(input);
        assert_eq!(was, Parsed::Diverges);
    }

    #[test]
    fn diverges() {
        let input = &char_slice("hi, mom");
        let was = diverge().parse(input);
        assert_eq!(was.clone().emits(), None);
        assert_eq!(was.into_option(), None);
    }

    #[test]
    fn optionality() {
        let input = &char_slice("hi, mom");

        let was = char('x').optionally().parse(input);
        assert_eq!(was.clone().emits(), Some(None));
        assert_eq!(
            was.into_option(),
            Some((Option::<char>::None, input.as_slice()))
        );
    }

    #[test]
    fn rep_sep() {
        let input = &char_slice("The quick brown fox jumps over the lazy dog,bastard");
        let word = such_that(|c| char::is_alphabetic(*c)).one_or_more();

        let word_input = &char_slice("dog,bastard");
        let was = word.parse(word_input);
        assert_eq!(was.emits(), Some(char_slice("dog")));

        let was = separated_by(word, char(' ').map(|_| ())).parse(input);
        let expected = vec![
            "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
        ]
        .into_iter()
        .map(char_slice)
        .collect::<Vec<_>>();

        assert_eq!(was.clone().emits(), Some(expected.clone()));
        assert_eq!(
            was.into_option(),
            Some((expected, char_slice(",bastard").as_slice()))
        )
    }

    #[test]
    fn skips() {
        let input = &char_slice("Hi, mom");
        let was = string("Hi")
            .skip_right(string(", "))
            .and_also(string("mom"))
            .parse(input);
        let expected = (char_slice("Hi"), char_slice("mom"));
        assert_eq!(was.clone().emits(), Some(expected));
        assert_eq!(
            was.into_option(),
            Some((
                (char_slice("Hi"), char_slice("mom")),
                char_slice("").as_slice()
            ))
        )
    }

    #[test]
    fn enclosed_within_test() {
        let input = &char_slice(" h");
        let ws = such_that(|c| char::is_whitespace(*c)).zero_or_more();

        let was = ws.skip_left(char('h')).parse(input);
        assert_eq!(was.emits(), Some('h'));

        let input = &char_slice(" [1, 2, 3, 4, 5]");
        let was = enclosed_within(
            ws.skip_left(char('[')),
            ws.skip_left(char(']')),
            separated_by(ws.skip_left(digit()), char(',').ignore()),
        )
        .parse(input);
        let expected = vec!['1', '2', '3', '4', '5'];

        assert_eq!(was.clone().emits(), Some(expected.clone()));
        assert_eq!(
            was.into_option(),
            Some((expected, char_slice("").as_slice()))
        );
    }
}
