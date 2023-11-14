### Sylt ###

This is a toy language that is a complete waste of time. It started out in response to @TsodingDaily presenting a way to implement regular expressions using FSM:s, apparently encountering them for the first time. 

I started out writing a parser and a lexer in his style, feeling the inspiration happening. I should write myself a parser combinator library on top of these newfound toys. I decided against this and started looking at the current state of ther art, which seems to be Nom and combine; neither of which caught my fancy. In fact: i could never get my parsers working quite right in combine and the gymnastics it required in the typing department are off putting.

So I wrote my own, instead, and as soon as I reached a criticle mass of useful combinators, I had a working JSON parser, and one that could tackle a miniscule toy language somewhat akin to the Rust basics. 

Parser combinators have always facinated me and I've written such a library in Scala and F# before - both of which are (probably?) here on GibHub.

Doing this in Rust has been a very different experience because, as it were, one must not write $language_a in $LanguageB because it tends to end in tears. Commonly, people will want to "Write Scala in Haskell" for instance; Scala handles that quite well but you shouldn't. The way you would do it in Haskell, Scala (and OCaml/ Standard ML, Lisp, etc) is to write combinators over a trivial parse function but Rust doesn't quite lend itself to this and you find yourself man handling it.

I decided instead to go with a Parser trait that applies to state structs describing (almost) every combinator. E.g.:

```F#
(* I have successfully parsed a thing, please emit a value.  *)
type ('a, 't) ParseResult = Emits of 'a * ('t list) | Diverges;;

(* Parses value a from a stream of tokens t. *)
type ('a, 't) Parser = 't list -> ParseResult<'a, 't>;;

let run p = p;;

(* Recognizes token t and emits it. *)
let token (p : 't -> bool) : Parser<'t, 't> =
    function (tok::remains) when p(tok) -> Emits (tok, remains) | _ -> Diverges;;

(* Maps any value a into a value b. E.g.: the number 427 into the text four 
   hundred and twenty seven, given the fortitude to express it. *)
let map (f : 'a -> 'b) (p : Parser<'a, 't>) : Parser<'b, 't> =
    p >> function Emits (a, remains) -> Emits (f a, remains) | _ -> Diverges;;

(* Expresses connecting any value a with a continuation of value b, enabling constructing
   an arbitrary sequence of heterogenously typed values. E.g.: parse a [, then a number, 
   then (optionally) a sequence of JSON values separated by ,:

   let arrayOpen = token (function tok -> tok = '[' | -> false );;
   let arrayClose = token (function tok -> tok = ']' | -> false );;
   let arraySep = token ( ... );;
   let rec jsonValue = ...;
   and jsonArray =
        arrayOpen
        |> bind (fun _ -> optionally (separatedBy jsonValue arraySep) jsonValue)
        |> bind (array -> map (fun _ -> array) arrayClose)
       
   tokenize "[1, 2, 3, 4]"
   |> run jsonValue 
   *)
let bind (f : 'a -> Parser<'b, 't>) (p : Parser<'a, 't>) : Parser<'a, 't> =
    run p >> function Emits (a, remains) -> f a remains | _ -> Diverges;;

```

This is not an F# project but looking at this, even as someone not quite familiar with F#, other MLs, or functional programming in general, it is apparent that there's a whole lot of functioning going on here. This is not how all F# looks, I merely chose it as the vessel to carry the functional approach to solving problems. Lots of "care" is taken to hide some of the parameter passing using currying because in doing so, the reader might focus more on the fact that these are abstract types expressing concepts, that we manipulate using combinators such that their final form might become a JSON parser, for instance. Functions are values.

```Rust
// I have successfully parsed a thing, please emit a thing. Unless I diverged in attempting to do so.
enum ParseResult<'a, A, T> {
    Emits(A, &'a [T]),
    Diverges,
}

// Expresses the capacity of whatever self is, that it can be instrumental in parsing a
// unit of In, producing a unit of Out, and the remains of In, in doing so
trait Parser {
    type In;
    type Out;

    fn parse<'a>(self, input: &'a [Self::In]) -> ParseResult<'a, Self::Out, Self::In>;

    fn map<F>(self, f: F) -> Map<F, Self::Out, Self::In> {
        Map(f, self, PhantomType)
    }
}

// An instrument expressing recognition of a value A. PhantomType helps rustc remember
// a piece of typing information from one point in the code stream to another. Don't
// worry about it.
fn token<F>(p: F) -> Token<F, A> {
    Token(p, PhantomType)
}

struct Token<F, A>(F, PhantomType<A>);

// Look: when you encounter a Token struct value, this is what I want you to do.
impl<F, A> Parser for Token<F, A> 
where
    F: FnOnce(&A) -> bool,
{
    type In = A;
    type Out = A;
    fn parse<'a>(self, input: &'a [Self::In]) -> ParseResult<'a, Self::Out, Self::In> {
        let Token(p, ..) = self;
        match input {
            [tok, remains @ ..] if p(tok) => ParseResult::Emits(tok, remains),
            _ => ParseResult::Diverges,
        }
    }
}

// Maps any value a into a value b. E.g.: the number 427 into the text four 
//   hundred and twenty seven, given the fortitude to express it.
struct Map<F, A, P>(F, PhantomType<A>, P);

impl<F, A, B, P> Parser for Map<F, A, P>
where 
    P: Parser<Out = A>,
{
    type In = P::In;
    type Out = B;

    fn parse<'a>(self, input: &'a [Self::In]) -> ParseResult<'a, Self::Out, Self::In> {
        let Map(f, inner, ..) = self;
        match self.parse(input) {
            ParseResult::Emits(a, remains) => ParseResult::Emits(f(a), remains),
            _ => ParseResult::Diverges,
        }
    }
}


// Parse an even number and transform it into a message telling us about this fact.
let parser = token(|x| x % 2 == 0).map(|x| format!("I have parsed {}", x));

```

