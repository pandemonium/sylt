use std::marker;

#[derive(Clone, Copy)]
struct Position {
    line: usize,
    column: usize,
}

impl Position {
    fn default() -> Self {
        Self { line: 1, column: 1 }
    }

    fn update_line(self, delta: usize) -> Self {
        Self {
            line: self.line + delta,
            ..self
        }
    }

    fn update_column(self, delta: usize) -> Self {
        Self {
            column: self.column + delta,
            ..self
        }
    }
}

// What can this do?
// An actual parser doesn't have access to this structure.
struct Error {
    expectations: Vec<String>,
}

#[derive(Clone, Copy)]
struct ParseState<'a, T> {
    remains: &'a [T],
    at: Position,
}

impl<'a, T> ParseState<'a, T> {
    pub fn new(input: &'a [T]) -> Self {
        Self {
            remains: input,
            at: Position::default(),
        }
    }

    pub fn position(&mut self) -> &mut Position {
        &mut self.at
    }

    pub fn can_advance(&self, by: usize) -> bool {
        self.remains.len() >= by
    }

    pub fn skip(self, by: usize) -> Self {
        Self {
            remains: &self.remains[by..],
            ..self
        }
    }
    pub fn take(&self, by: usize) -> &'a [T] {
        &self.remains[..by]
    }

    pub fn head(&self) -> Option<&T> {
        if let &[ref head, ..] = self.remains {
            Some(&head)
        } else {
            None
        }
    }
}

struct ParseResult<'a, T, A> {
    state: ParseState<'a, T>,
    parsed: Option<A>,
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

trait Parser {
    type Input: Clone;
    type Output;

    fn parse<'a>(
        self,
        input: ParseState<'a, Self::Input>,
    ) -> ParseResult<'a, Self::Input, Self::Output>;

    fn map<F, B>(self, f: F) -> Map<F, Self, B>
    where
        F: FnOnce(Self::Output) -> B,
        Self: Sized,
    {
        Map(f, self, marker::PhantomData)
    }

    fn flat_map<F, Q>(self, f: F) -> FlatMap<F, Self, Q>
    where
        Q: Parser,
        F: FnOnce(Self::Output) -> Q,
        Self: Sized,
    {
        FlatMap(f, self, marker::PhantomData)
    }

    fn and_also<P>(self, rhs: P) -> AndAlso<Self, P>
    where
        Self: Sized,
    {
        AndAlso(self, rhs)
    }

    fn accept<F>(self, p: F) -> Accept<Self, F>
    where
        F: FnOnce(&Self::Output) -> bool,
        Self: Sized,
    {
        Accept(self, p)
    }
}

fn take<T>(length: usize) -> Take<T> {
    Take(length, marker::PhantomData)
}

struct Take<T>(usize, marker::PhantomData<T>);

impl<T> Parser for Take<T>
where
    T: Clone,
{
    type Input = T;
    type Output = Vec<T>;

    fn parse<'a>(
        self,
        input: ParseState<'a, Self::Input>,
    ) -> ParseResult<'a, Self::Input, Self::Output> {
        let Take(length, ..) = self;
        if input.can_advance(length) {
            let token = input.take(length);
            ParseResult {
                state: input.skip(length),
                parsed: Some(token.to_vec()),
            }
        } else {
            todo!()
        }
    }
}

struct Map<F, P, B>(F, P, marker::PhantomData<B>);

impl<F, P, B> Parser for Map<F, P, B>
where
    P: Parser,
    F: FnOnce(P::Output) -> B,
{
    type Input = P::Input;
    type Output = B;

    fn parse<'a>(
        self,
        input: ParseState<'a, Self::Input>,
    ) -> ParseResult<'a, Self::Input, Self::Output> {
        let Map(f, inner, ..) = self;
        inner.parse(input).map(f)
    }
}

struct FlatMap<F, P, Q>(F, P, marker::PhantomData<Q>);

impl<F, P, Q> Parser for FlatMap<F, P, Q>
where
    P: Parser,
    Q: Parser<Input = P::Input>,
    F: FnOnce(P::Output) -> Q,
{
    type Input = P::Input;
    type Output = Q::Output;

    fn parse<'a>(
        self,
        input: ParseState<'a, Self::Input>,
    ) -> ParseResult<'a, Self::Input, Self::Output> {
        let FlatMap(f, inner, ..) = self;
        let res = inner.parse(input.clone()).map(f);

        if let Some(q) = res.parsed {
            q.parse(res.state)
        } else {
            ParseResult::balked(input)
        }
    }
}

struct AndAlso<P, Q>(P, Q);

impl<P, Q> Parser for AndAlso<P, Q>
where
    P: Parser,
    P::Output: Clone,
    Q: Parser<Input = P::Input>,
{
    type Input = Q::Input;
    type Output = (P::Output, Q::Output);

    fn parse<'a>(
        self,
        input: ParseState<'a, Self::Input>,
    ) -> ParseResult<'a, Self::Input, Self::Output> {
        let AndAlso(p, q) = self;
        p.flat_map(|x| q.map(|y| (x, y))).parse(input)
    }
}

fn such_that<F, T>(f: F) -> SuchThat<F, T>
where
    F: FnOnce(&T) -> bool,
{
    SuchThat(f, marker::PhantomData)
}

struct SuchThat<F, T>(F, marker::PhantomData<T>);

impl<F, T> Parser for SuchThat<F, T>
where
    F: FnOnce(&T) -> bool,
    T: Clone,
{
    type Input = T;
    type Output = T;

    fn parse<'a>(
        self,
        input: ParseState<'a, Self::Input>,
    ) -> ParseResult<'a, Self::Input, Self::Output> {
        let SuchThat(p, ..) = self;
        match input.head() {
            Some(head) if p(head) => {
                let head = head.clone();
                ParseResult::accepted(input.skip(1), head)
            }
            _otherwise => ParseResult::balked(input),
        }
    }
}

struct Accept<P, F>(P, F);

impl<P, F> Parser for Accept<P, F>
where
    P: Parser,
    P::Output: Clone,
    F: FnOnce(&P::Output) -> bool,
{
    type Input = P::Input;
    type Output = P::Output;

    fn parse<'a>(
        self,
        input: ParseState<'a, Self::Input>,
    ) -> ParseResult<'a, Self::Input, Self::Output> {
        let Accept(inner, p) = self;
        let res = inner.parse(input);

        match res.parsed {
            Some(ref x) if p(x) => ParseResult::accepted(res.state, x.clone()),
            _otherwise => ParseResult::balked(res.state),
        }
    }
}

type ParserBox<I, O> = Box<dyn Parser<Input = I, Output = O>>;

fn char(c: char) -> ParserBox<char, char> {
    Box::new(such_that(move |x| x == &c))
}

#[cfg(test)]
mod tests {
    use crate::kombi::such_that;

    use super::{take, ParseState, Parser};
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
    fn accepts() {
        let input = "Hi, mom".chars().collect::<Vec<_>>();
        let was = take(3)
            .accept(|x| x.len() == 3)
            .map(|xs| xs.iter().collect::<String>())
            .parse(ParseState::new(&input));
        assert_eq!(was.into_option(), Some("Hi,".into()))
    }
}
