use std::marker;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    // Make configurable so that the behaviour
    //  of CR, LF and tabs can be controlled.
    line: usize,
    column: usize,
}

impl Position {
    pub fn default() -> Self {
        Self { line: 1, column: 1 }
    }

    pub fn carriage_return(self) -> Self {
        Self {
            line: self.line + 1,
            column: 1,
            ..self
        }
    }

    pub fn line_feed(self) -> Self {
        Self {
            line: self.line + 1,
            column: 1,
            ..self
        }
    }

    pub fn move_right(self) -> Self {
        Self {
            column: self.column + 1,
            ..self
        }
    }
}

// What can this do?
// An actual parser doesn't have access to this structure.
struct Error {
    expectations: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct ParseState<'a, T> {
    pub token: &'a [T],
    pub remains: &'a [T],
    pub at: Position,
}

impl<'a, T> ParseState<'a, T> {
    pub fn new(input: &'a [T]) -> Self {
        Self {
            token: &input[..0],
            remains: input,
            at: Position::default(),
        }
    }

    pub fn map_position<F>(self, f: F) -> Self
    where
        F: FnOnce(Position) -> Position,
    {
        Self {
            at: f(self.at),
            ..self
        }
    }

    pub fn can_advance(&self, by: usize) -> bool {
        self.remains.len() >= by
    }

    pub fn advance(self, by: usize) -> Self {
        Self {
            token: &self.remains[..by],
            remains: &self.remains[by..],
            ..self
        }
    }

    pub fn token(&self) -> &'a [T] {
        &self.token
    }

    pub fn peek(&self) -> Option<&T> {
        if let &[ref head, ..] = self.remains {
            Some(&head)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct ParseResult<'a, T, A> {
    pub state: ParseState<'a, T>,
    pub parsed: Option<A>,
}

impl<'a, T, A> ParseResult<'a, T, A> {
    pub fn balked(state: ParseState<'a, T>) -> Self {
        Self {
            state,
            parsed: None,
        }
    }

    pub fn accepted(state: ParseState<'a, T>, returns: A) -> Self {
        Self {
            state,
            parsed: Some(returns),
        }
    }

    pub fn into_option(self) -> Option<A> {
        self.parsed
    }

    pub fn map<F, B>(self, f: F) -> ParseResult<'a, T, B>
    where
        F: FnOnce(A) -> B,
    {
        ParseResult {
            state: self.state,
            parsed: self.parsed.map(f),
        }
    }
}

pub trait Parser: Clone {
    type In: Clone;
    type Out: Clone;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out>;

    fn map<F, B>(self, f: F) -> Map<F, Self, B>
    where
        F: FnOnce(Self::Out) -> B,
        Self: Sized,
    {
        Map(f, self, marker::PhantomData)
    }

    fn flat_map<F, Q>(self, f: F) -> FlatMap<F, Self, Q>
    where
        Q: Parser,
        F: FnOnce(Self::Out) -> Q,
        Self: Sized,
    {
        FlatMap(f, self, marker::PhantomData)
    }

    fn filter_map<F, B>(self, f: F) -> FilterMap<F, Self, B>
    where
        F: FnOnce(Self::Out) -> Option<B>,
        Self: Sized,
    {
        FilterMap(f, self, marker::PhantomData)
    }

    fn and_also<P>(self, rhs: P) -> AndAlso<Self, P>
    where
        Self: Sized,
    {
        AndAlso(self, rhs)
    }

    fn or_else<P>(self, rhs: P) -> OrElse<Self, P>
    where
        Self: Sized,
    {
        OrElse(self, rhs)
    }

    fn when<F>(self, p: F) -> When<Self, F>
    where
        F: FnOnce(&Self::Out) -> bool,
        Self: Sized,
    {
        When(self, p)
    }

    fn zero_or_more(self) -> ZeroOrMore<Self>
    where
        Self: Sized,
    {
        ZeroOrMore(self)
    }

    fn one_or_more(self) -> OneOrMore<Self>
    where
        Self: Sized,
    {
        OneOrMore(self)
    }

    fn optionally(self) -> Optionally<Self>
    where
        Self: Sized,
    {
        Optionally(self)
    }

    fn skip_left<Q>(self, rhs: Q) -> SkipLeft<Self, Q>
    where
        Self: Sized,
    {
        SkipLeft(self, rhs)
    }

    fn skip_right<Q>(self, rhs: Q) -> SkipRight<Self, Q>
    where
        Self: Sized,
    {
        SkipRight(self, rhs)
    }
}

#[derive(Clone, Debug)]
pub struct Positions<P>(P);

pub trait Positioned<P> {
    fn with_positions(self) -> Positions<P>;
}

impl<P> Positioned<P> for P
where
    P: Parser<In = char>,
{
    fn with_positions(self) -> Positions<P> {
        Positions(self)
    }
}

impl<P> Parser for Positions<P>
where
    P: Parser<In = char>,
{
    type In = P::In;
    type Out = P::Out;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let Positions(inner) = self;
        let mut result = inner.parse(input);

        if result.parsed.is_some() {
            let token = result.state.token();
            result.state = token.iter().fold(result.state, |st, x| {
                st.map_position(|prompt| match x {
                    '\r' => prompt.carriage_return(),
                    '\n' => prompt.line_feed(),
                    _otherwise => prompt.move_right(),
                })
            });
        }

        result
    }
}

pub fn take<T>(length: usize) -> Take<T> {
    Take(length, marker::PhantomData)
}

#[derive(Clone, Debug)]
pub struct Take<T>(usize, marker::PhantomData<T>);

impl<T> Parser for Take<T>
where
    T: Clone,
{
    type In = T;
    type Out = Vec<T>;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let Take(length, ..) = self;
        if input.can_advance(length) {
            let input = input.advance(length);
            let token = input.token();
            ParseResult::accepted(input, token.to_vec())
        } else {
            ParseResult::balked(input)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Map<F, P, B>(F, P, marker::PhantomData<B>);

impl<F, P, B> Parser for Map<F, P, B>
where
    P: Parser,
    F: FnOnce(P::Out) -> B + Clone,
    B: Clone,
{
    type In = P::In;
    type Out = B;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let Map(f, inner, ..) = self;
        inner.parse(input).map(f)
    }
}

#[derive(Clone, Debug)]
pub struct FlatMap<F, P, Q>(F, P, marker::PhantomData<Q>);

impl<F, P, Q> Parser for FlatMap<F, P, Q>
where
    P: Parser,
    Q: Parser<In = P::In>,
    F: FnOnce(P::Out) -> Q + Clone,
{
    type In = P::In;
    type Out = Q::Out;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let FlatMap(f, inner, ..) = self;
        let ParseResult { state, parsed } = inner.parse(input.clone()).map(f);

        parsed
            .map(|q| q.parse(state))
            .unwrap_or_else(|| ParseResult::balked(input))
    }
}

#[derive(Clone, Debug)]
pub struct FilterMap<F, P, B>(F, P, marker::PhantomData<B>);

impl<F, P, B> Parser for FilterMap<F, P, B>
where
    P: Parser,
    F: FnOnce(P::Out) -> Option<B> + Clone,
    B: Clone,
{
    type In = P::In;
    type Out = B;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let FilterMap(f, inner, ..) = self;
        if let ParseResult {
            state,
            parsed: Some(Some(x)),
        } = inner.map(f).parse(input.clone())
        {
            ParseResult::accepted(state, x)
        } else {
            ParseResult::balked(input)
        }
    }
}

#[derive(Clone, Debug)]
pub struct AndAlso<P, Q>(P, Q);

impl<P, Q> Parser for AndAlso<P, Q>
where
    P: Parser,
    Q: Parser<In = P::In>,
{
    type In = Q::In;
    type Out = (P::Out, Q::Out);

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let AndAlso(p, q) = self;
        p.flat_map(|x| q.map(|y| (x, y))).parse(input)
    }
}

#[derive(Clone, Debug)]
pub struct OrElse<P, Q>(P, Q);

impl<P, Q> Parser for OrElse<P, Q>
where
    P: Parser,
    Q: Parser<In = P::In, Out = P::Out>,
{
    type In = Q::In;
    type Out = Q::Out;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let OrElse(p, q) = self;

        if let ParseResult {
            state,
            parsed: Some(x),
        } = p.parse(input.clone())
        {
            ParseResult::accepted(state, x)
        } else {
            q.parse(input)
        }
    }
}

pub fn such_that<F, T>(f: F) -> SuchThat<F, T>
where
    F: FnOnce(&T) -> bool,
{
    SuchThat(f, marker::PhantomData)
}

#[derive(Clone, Debug)]
pub struct SuchThat<F, T>(F, marker::PhantomData<T>);

impl<F, T> Parser for SuchThat<F, T>
where
    F: FnOnce(&T) -> bool + Clone,
    T: Clone,
{
    type In = T;
    type Out = T;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let SuchThat(p, ..) = self;

        match input.peek() {
            Some(head) if p(head) => {
                let head = head.clone();
                ParseResult::accepted(input.advance(1), head)
            }
            _otherwise => ParseResult::balked(input),
        }
    }
}

#[derive(Clone, Debug)]
pub struct When<P, F>(P, F);

impl<P, F> Parser for When<P, F>
where
    P: Parser,
    F: FnOnce(&P::Out) -> bool + Clone,
{
    type In = P::In;
    type Out = P::Out;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let When(inner, p) = self;
        let ParseResult { state, parsed } = inner.parse(input);

        match parsed {
            Some(ref x) if p(x) => ParseResult::accepted(state, x.clone()),
            _otherwise => ParseResult::balked(state),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZeroOrMore<P>(P);

impl<P> Parser for ZeroOrMore<P>
where
    P: Parser,
{
    type In = P::In;
    type Out = Vec<P::Out>;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let ZeroOrMore(inner) = self;
        let mut output = vec![];
        let mut cursor = input;

        let remains = loop {
            let ParseResult { state, parsed } = inner.clone().parse(cursor);
            if let Some(x) = parsed {
                output.push(x);
                cursor = state
            } else {
                break state;
            }
        };

        ParseResult::accepted(remains, output)
    }
}

pub fn separated_by<P, S>(p: P, s: S) -> SeparatedBy<P, S> {
    SeparatedBy(p, s)
}

#[derive(Clone, Debug)]
pub struct SeparatedBy<P, S>(P, S);

impl<P, S> Parser for SeparatedBy<P, S>
where
    P: Parser,
    S: Parser<In = P::In>,
{
    type In = S::In;
    type Out = Vec<P::Out>;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let SeparatedBy(inner, separator) = self;
        inner
            .clone()
            .and_also(separator.and_also(inner).map(snd).zero_or_more())
            .map(|(head, mut tail)| {
                tail.insert(0, head);
                tail
            })
            .or_else(empty(vec![]))
            .parse(input)
    }
}

pub fn empty<I, O>(x: O) -> Empty<I, O> {
    Empty(x, marker::PhantomData)
}

#[derive(Clone, Debug)]
pub struct Empty<I, O>(O, marker::PhantomData<I>);

impl<I, O> Parser for Empty<I, O>
where
    I: Clone,
    O: Clone,
{
    type In = I;
    type Out = O;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let Empty(out, ..) = self;
        ParseResult::accepted(input, out)
    }
}

#[derive(Clone, Debug)]
pub struct Optionally<P>(P);

impl<P> Parser for Optionally<P>
where
    P: Parser,
{
    type In = P::In;
    type Out = Option<P::Out>;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let Optionally(inner) = self;
        inner.map(Some).or_else(empty(None)).parse(input)
    }
}

#[derive(Clone, Debug)]
pub struct OneOrMore<P>(P);

impl<P> Parser for OneOrMore<P>
where
    P: Parser,
{
    type In = P::In;
    type Out = Vec<P::Out>;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let OneOrMore(inner) = self;
        inner
            .clone()
            .and_also(inner.zero_or_more())
            .map(|(head, mut tail)| {
                tail.insert(0, head);
                tail
            })
            .parse(input)
    }
}

#[derive(Clone, Debug)]
pub struct SkipLeft<P, Q>(P, Q);

impl<P, Q> Parser for SkipLeft<P, Q>
where
    P: Parser,
    Q: Parser<In = P::In>,
{
    type In = P::In;
    type Out = Q::Out;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let SkipLeft(p, q) = self;
        p.and_also(q).map(snd).parse(input)
    }
}

#[derive(Clone, Debug)]
pub struct SkipRight<P, Q>(P, Q);

impl<P, Q> Parser for SkipRight<P, Q>
where
    P: Parser,
    Q: Parser<In = P::In>,
{
    type In = P::In;
    type Out = P::Out;

    fn parse<'a>(self, input: ParseState<'a, Self::In>) -> ParseResult<'a, Self::In, Self::Out> {
        let SkipRight(p, q) = self;
        p.and_also(q).map(fst).parse(input)
    }
}

#[inline]
pub fn fst<A, B>(tuple: (A, B)) -> A {
    tuple.0
}

#[inline]
pub fn snd<A, B>(tuple: (A, B)) -> B {
    tuple.1
}

pub fn char(c: char) -> impl Parser<In = char, Out = char> {
    such_that(move |x| x == &c)
}

pub fn string(literal: &str) -> impl Parser<In = char, Out = Vec<char>> {
    let literal = literal.chars().collect::<Vec<_>>();
    literally::<char>(&literal).with_positions()
}

pub fn enclosed_within<P, Q, R>(open: P, close: R, body: Q) -> impl Parser<In = P::In, Out = Q::Out>
where
    P: Parser,
    Q: Parser<In = P::In>,
    R: Parser<In = Q::In>,
{
    open.skip_left(body).skip_right(close)
}

pub fn literally<T>(literal: &[T]) -> impl Parser<In = T, Out = Vec<T>>
where
    T: Clone + PartialEq,
{
    let literal = literal.to_vec();
    take(literal.len()).when(move |xs| xs.as_slice() == literal)
}

pub fn one_of(choice: &[char]) -> impl Parser<In = char, Out = char> {
    let choice = choice.to_vec();
    such_that(move |c| choice.contains(c))
}

#[macro_export]
macro_rules! match_map {
    ($expression:expr, $( $pattern:pat )+ $( if $guard: expr )? => $ret:expr) => {
        match $expression {
            $( $pattern )|+ $( if $guard )? => Some($ret),
            _ => None
        }
    }
}

#[macro_export]
macro_rules! expect {
    ( $($pat:pat $(if $guard:expr)? => $ret:expr),* ) => {
         filter_map(move |x| match x {
            $(
                $pat $(if $guard)? => Some($ret),
            )*
            _ => None,
         })
    };
}

pub fn filter_map<T, U, F>(p: F) -> impl Parser<In = T, Out = U>
where
    F: FnOnce(&T) -> Option<U> + Clone,
    T: Clone,
    U: Clone,
{
    take::<T>(1).filter_map(|xs| xs.first().and_then(p))
}

#[cfg(test)]
mod tests {
    use super::{
        char, enclosed_within, one_of, string, such_that, take, ParseState, Parser, Positioned,
    };
    use std::vec;

    #[test]
    fn takes() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = take(3).parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some(vec!['H', 'i', ',']));
    }

    #[test]
    fn maps() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = take(3).map(|x| x.len()).parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some(3));
    }

    #[test]
    fn flat_maps() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = take(3)
            .flat_map(|x| take(4).map(|y| (x, y)))
            .parse(ParseState::new(&input));
        assert_eq!(
            was.into_option(),
            Some((vec!['H', 'i', ','], vec![' ', 'm', 'o', 'm']))
        );
    }

    #[test]
    fn and_also() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = take(3).and_also(take(4)).parse(ParseState::new(&input));
        assert_eq!(
            was.into_option(),
            Some((vec!['H', 'i', ','], vec![' ', 'm', 'o', 'm']))
        );

        let was = such_that(|x: &char| x == &'a')
            .and_also(take(4))
            .parse(ParseState::new(&input));
        assert_eq!(was.into_option(), None,);
    }

    #[test]
    fn or_else() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();

        let was = take(3).or_else(take(4)).parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some(vec!['H', 'i', ',']));

        let was = such_that(|x| x == &'a')
            .or_else(char('H'))
            .parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some('H'));
    }

    #[test]
    fn such_thats() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();

        let was = such_that(|x| char::is_whitespace(*x)).parse(ParseState::new(&input));
        assert_eq!(was.into_option(), None,);

        let was = such_that(|x| x == &'H').parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some('H'),);
    }

    #[test]
    fn whens() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = take(3)
            .when(|x| x.len() == 3)
            .map(|xs| xs.iter().collect::<String>())
            .parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some("Hi,".into()))
    }

    #[test]
    fn zero_or_more() {
        let input = "aaaaab".chars().collect::<Vec<_>>();
        let was = char('a')
            .zero_or_more()
            .map(|xs| xs.iter().collect::<String>())
            .parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some("aaaaa".into()))
    }

    #[test]
    fn one_ofs() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = one_of(&['H', 'a']).parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some('H'));
    }

    #[test]
    fn strings() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = string("Hi,")
            .and_also(string(" mom"))
            .map(|(mut p, mut q)| {
                p.append(&mut q);
                p.iter().collect::<String>()
            })
            .parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some("Hi, mom".into()));
    }

    #[test]
    fn enclosed_within_test() {
        let input = r#""Hi, mom""#.chars().collect::<Vec<_>>();
        let was = enclosed_within(
            char('"'),
            char('"'),
            such_that(|c| c != &'"').zero_or_more(),
        )
        .map(|xs| xs.iter().collect::<String>())
        .parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some("Hi, mom".into()));
    }

    #[test]
    fn positions() {
        let input = r#"
        
        "#
        .chars()
        .collect::<Vec<_>>();
        let was = such_that(|c| true)
            .zero_or_more()
            .parse(ParseState::new(&input));

        println!("{:?}", was.state);
    }
}
