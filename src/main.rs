use sylt::{
    ast::*,
    runtime::{
        interpreter,
        intrinsics::{artithmetic::operator, io},
    }, lexer, syntax,
};

fn main() {
    let source = r#"
{
    println("Hello, world");
    while True {
        if 12/34*56 + "Kalle" {
            let foo = 124 + 42.5 / 10 * 0.1;
        } else {
            save(427, "data.txt");
        }
    }
}    
    "#;

    let program = syntax::analyze(&source.chars().collect::<Vec<_>>());
    println!("{:?}", program);
}
