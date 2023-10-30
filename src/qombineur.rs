use std::marker;

#[derive(Clone, PartialEq, Debug)]
enum Parsed<'a, A, Input> {
    Emit(A, &'a [Input]),
    Diverged,
}

impl<'a, A, Input> Parsed<'a, A, Input>
//where
//    Input: Clone,
{
    fn map<F, B>(self, f: F) -> Parsed<'a, B, Input>
    where
        F: FnOnce(A) -> B,
    {
        match self {
            Self::Emit(x, remains) => Parsed::Emit(f(x), remains),
            Self::Diverged => Parsed::Diverged,
        }
    }

    fn flat_map<F, B>(self, f: F) -> Parsed<'a, B, Input>
    where
        F: FnOnce(A, &'a [Input]) -> Parsed<'a, B, Input>,
    {
        match self {
            Parsed::Emit(x, remains) => f(x, remains),
            Parsed::Diverged => Parsed::Diverged,
        }
    }

    fn emits(self) -> Option<A> {
        if let Self::Emit(x, _) = self {
            Some(x)
        } else {
            None
        }
    }

    fn into_option(self) -> Option<(A, &'a [Input])> {
        if let Self::Emit(x, remains) = self {
            Some((x, remains))
        } else {
            None
        }
    }
}

trait Parsimonious<A>: Clone + Sized {
    type Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token>;

    fn parse_phrase<'a>(self, input: &'a [Self::Token]) -> Option<A> {
        self.parse(input).into_option().and_then(
            |(x, remains)| {
                if remains.is_empty() {
                    Some(x)
                } else {
                    None
                }
            },
        )
    }

    fn map<F, B>(self, f: F) -> Map<F, A, Self>
    where
        F: FnOnce(A) -> B,
    {
        Map {
            inner: self,
            f,
            _inner_output: Default::default(),
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
            _inner_output: Default::default(),
        }
    }

    fn or_else<Q>(self, alternative: Q) -> OrElse<Self, Q>
    where
        Q: Parsimonious<A, Token = Self::Token>,
    {
        OrElse {
            lhs: self,
            rhs: alternative,
        }
    }

    fn and_also<B, Q>(self, sequeteur: Q) -> AndAlso<Self, Q>
    where
        Q: Parsimonious<B, Token = Self::Token>,
    {
        AndAlso {
            lhs: self,
            rhs: sequeteur,
        }
    }

    fn optionally(self) -> Optionally<Self> {
        Optionally(self)
    }

    fn zero_or_more(self) -> ZeroOrMore<Self> {
        ZeroOrMore(self)
    }

    fn one_or_more(self) -> OneOrMore<Self> {
        OneOrMore(self)
    }
}

fn diverge<A, T>() -> Diverged<A, T> {
    Diverged(Default::default(), Default::default())
}

#[derive(Clone, Copy)]
struct Diverged<A, T>(marker::PhantomData<A>, marker::PhantomData<T>);

impl<A, T> Parsimonious<A> for Diverged<A, T>
where
    A: Clone,
    T: Clone,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token> {
        Parsed::Diverged
    }
}

#[derive(Clone, Copy)]
struct OneOrMore<P>(P);

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
            .optionally()
            .parse(input)
            .flat_map(|x, remains| {
                if let Some(x) = x {
                    inner.zero_or_more().parse(remains).map(|mut xs| {
                        xs.insert(0, x);
                        xs
                    })
                } else {
                    Parsed::Diverged
                }
            })
    }
}

#[derive(Clone, Copy)]
struct Optionally<P>(P);

impl<P, A> Parsimonious<Option<A>> for Optionally<P>
where
    P: Parsimonious<A>,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Option<A>, Self::Token> {
        let Optionally(inner) = self;

        inner.parse(input).map(Some)
    }
}

#[derive(Clone, Copy)]
struct ZeroOrMore<P>(P);

impl<P, A> Parsimonious<Vec<A>> for ZeroOrMore<P>
where
    P: Parsimonious<A>,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<A>, Self::Token> {
        let ZeroOrMore(inner) = &self;
        let mut cursor = input;
        let mut result: Vec<A> = vec![];

        // This can probably be re-written in terms of optional, flat_map and recursion.
        loop {
            match inner.clone().parse(cursor) {
                Parsed::Emit(x, remains) => {
                    result.push(x);
                    cursor = remains;
                }
                Parsed::Diverged => break,
            }
        }

        Parsed::Emit(result, cursor)
    }
}

#[derive(Clone, Copy)]
struct Map<F, A, P> {
    inner: P,
    f: F,
    _inner_output: marker::PhantomData<A>,
}

impl<F, A, B, P> Parsimonious<B> for Map<F, A, P>
where
    P: Parsimonious<A>,
    F: FnOnce(A) -> B + Clone,
    A: Clone,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, B, Self::Token> {
        self.inner.parse(input).map(self.f)
    }
}

#[derive(Clone, Copy)]
struct FlatMap<F, A, P> {
    inner: P,
    f: F,
    _inner_output: marker::PhantomData<A>,
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

fn empty<A, T>(produces: A) -> Empty<A, T>
where
    A: Clone,
{
    Empty {
        produces,
        _token_type: Default::default(),
    }
}

#[derive(Clone, Copy)]
struct Empty<A, T> {
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
        Parsed::Emit(self.produces.clone(), input)
    }
}

#[derive(Clone, Copy, Debug)]
struct Any<T>(marker::PhantomData<T>);

fn any<T>() -> Any<T> {
    Any(Default::default())
}

impl<T> Parsimonious<T> for Any<T>
where
    T: Clone, /* Copy instead? */
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, T, Self::Token> {
        if !input.is_empty() {
            Parsed::Emit(input[0].clone(), &input[1..])
        } else {
            Parsed::Diverged
        }
    }
}

fn such_that<T, F>(predicate: F) -> SuchThat<F, T>
where
    F: Fn(&T) -> bool,
{
    SuchThat {
        predicate,
        _token_type: Default::default(),
    }
}

#[derive(Clone, Copy)]
struct SuchThat<F, T> {
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
            [head, tail @ ..] if (self.predicate)(head) => Parsed::Emit(input[0].clone(), tail),
            _otherwise => Parsed::Diverged,
        }
    }
}

fn this<T>(token: T) -> impl Parsimonious<T, Token = T>
where
    T: PartialEq + Clone,
{
    such_that(move |x| x == &token)
}

fn char(c: char) -> impl Parsimonious<char, Token = char> {
    this(c)
}

fn take<T>(i: usize) -> Take<T> {
    Take(i, Default::default())
}

#[derive(Clone, Copy)]
struct Take<T>(usize, marker::PhantomData<T>);

impl<T> Parsimonious<Vec<T>> for Take<T>
where
    T: Clone,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<T>, Self::Token> {
        let Take(length, ..) = self;
        if input.len() >= length {
            Parsed::Emit(
                input[0..length].into_iter().cloned().collect::<Vec<_>>(),
                &input[length..],
            )
        } else {
            Parsed::Diverged
        }
    }
}

fn string(literal: &str) -> LiteralMatch<char> {
    LiteralMatch {
        subject: literal.chars().collect(),
    }
}

fn literal<T>(literal: &[T]) -> LiteralMatch<T>
where
    T: Clone,
{
    LiteralMatch {
        subject: literal.to_vec(),
    }
}

#[derive(Clone)]
struct LiteralMatch<T> {
    subject: Vec<T>,
}

impl<T> Parsimonious<Vec<T>> for LiteralMatch<T>
where
    T: Clone + PartialEq,
{
    type Token = T;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, Vec<T>, Self::Token> {
        let token_length = self.subject.len();
        if input.len() > token_length && self.subject.into_iter().zip(input).all(|(p, q)| &p == q) {
            Parsed::Emit(input[0..token_length].to_vec(), &input[token_length..])
        } else {
            Parsed::Diverged
        }
    }
}

#[derive(Clone, Copy)]
struct OrElse<P, Q> {
    lhs: P,
    rhs: Q,
}

impl<A, P, Q, T> Parsimonious<A> for OrElse<P, Q>
where
    P: Parsimonious<A, Token = T>,
    Q: Parsimonious<A, Token = P::Token>,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, A, Self::Token> {
        if let return_value @ Parsed::Emit(..) = self.lhs.parse(input) {
            return_value
        } else {
            self.rhs.parse(input)
        }
    }
}

#[derive(Clone, Copy)]
struct AndAlso<P, Q> {
    lhs: P,
    rhs: Q,
}

impl<A, B, P, Q> Parsimonious<(A, B)> for AndAlso<P, Q>
where
    P: Parsimonious<A>,
    Q: Parsimonious<B, Token = P::Token>,
{
    type Token = P::Token;

    fn parse<'a>(self, input: &'a [Self::Token]) -> Parsed<'a, (A, B), Self::Token> {
        self.lhs
            .parse(input)
            .flat_map(|p, remains| self.rhs.parse(remains).map(|q| (p, q)))
    }
}

#[cfg(test)]
mod tests {
    use super::{any, char, empty, string, such_that, take, this, Parsed, Parsimonious};

    #[test]
    fn empty_parse() {
        let input = &"hi, mom".chars().collect::<Vec<_>>();
        let was = empty(427).parse(input);
        assert_eq!(was.clone().emits(), Some(427));
        assert_eq!(was.into_option(), Some((427, input.as_slice())));
    }

    #[test]
    fn any_parse() {
        let input = &"hi, mom".chars().collect::<Vec<_>>();
        let was = any().parse(input);
        assert_eq!(was.clone().emits(), Some('h'));
        assert_eq!(was.into_option(), Some(('h', &input.as_slice()[1..])));
    }

    #[test]
    fn or_else() {
        let input = &"hi, mom".chars().collect::<Vec<_>>();
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
        let input = &"hi, mom".chars().collect::<Vec<_>>();
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
            Some((('h', 'x'), &input.as_slice()[2..]))
        );

        let was = empty('x').and_also(any()).parse(input);
        assert_eq!(was.clone().emits(), Some(('x', 'h')));
        assert_eq!(
            was.into_option(),
            Some((('x', 'h'), &input.as_slice()[2..]))
        );

        let was = any().and_also(empty('x')).parse(empty_input);
        assert_eq!(was, Parsed::Diverged);
    }

    #[test]
    fn such_that_test() {
        let input = &"hi, mom".chars().collect::<Vec<_>>();
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
        let input = &"hi, mom".chars().collect::<Vec<_>>();
        let was = char('h')
            .and_also(this('i'))
            .and_also(such_that(|x| x == &','))
            .parse(input);

        assert_eq!(was.clone().emits(), Some((('h', 'i'), ',')));
        assert_eq!(was.into_option(), Some(((('h', 'i'), ','), &input[3..])));
    }

    #[test]
    fn map() {
        let input = &"hi, mom".chars().collect::<Vec<_>>();
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
        let input = &"hi, mom".chars().collect::<Vec<_>>();
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
        let input = &"hi, mom".chars().collect::<Vec<_>>();
        let was = take(3).parse(input);

        assert_eq!(was.clone().emits(), Some(vec!['h', 'i', ',']));
        assert_eq!(was.into_option(), Some((vec!['h', 'i', ','], &input[3..])));
    }

    #[test]
    fn strings() {
        let input = &"hi, mom".chars().collect::<Vec<_>>();
        let p = "hi, ";
        let q = "mom";

        let was = string(p).parse(input);
        assert_eq!(was.clone().emits(), Some(p.chars().collect()));
        assert_eq!(was.into_option(), Some((p.chars().collect(), &input[4..])));

        let was = string("hi, ").and_also(string("mom")).parse(input);
        assert_eq!(
            was.clone().emits(),
            Some((p.chars().collect(), q.chars().collect(),))
        );
        assert_eq!(
            was.into_option(),
            Some((
                (p.chars().collect(), q.chars().collect(),),
                "".chars().collect::<Vec<_>>().as_slice()
            ))
        )
    }

    #[test]
    fn zero_or_more() {
        let input = &"aaaaabbbbc".chars().collect::<Vec<_>>();
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
                "".chars().collect::<Vec<_>>().as_slice()
            ))
        )
    }

    #[test]
    fn one_or_more() {
        let input = &"aaaaabbbbc".chars().collect::<Vec<_>>();
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
                "".chars().collect::<Vec<_>>().as_slice()
            ))
        );

        let was = q.parse(input);
        assert_eq!(was, Parsed::Diverged);
    }
}
