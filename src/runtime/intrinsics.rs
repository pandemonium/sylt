use artithmetic::operator;
use crate::{ast, runtime::interpreter};

pub fn invoke_intrinsic_function(
    function: &ast::IntrinsicFunctionDef,
    arguments: &[ast::Constant],
) -> Result<ast::Constant, interpreter::Error> {
    // Really find a way to store a function parameter
    if operator::in_prefix(&function.name()) {
        Ok(operator::apply_by_name(
            function.name().local_name(),
            arguments,
        )?)
    } else if io::in_prefix(&function.name()) {
        Ok(io::apply_by_name(function.name().local_name(), arguments)?)
    } else {
        todo!()
    }
}

pub mod artithmetic {
    pub mod operator {
        use crate::{
            ast::{
                self,
                Constant::{self, *},
                Operator::{self, *},
            },
            runtime::interpreter::{self, Error},
        };

        static PREFIX: &[&str] = &["builtins", "arithmetic", "operator"];

        pub fn in_prefix(name: &ast::Name) -> bool {
            PREFIX == name.scope_path
        }

        pub fn qualified_name(local_name: &str) -> ast::Name {
            ast::Name::intrinsic("arithmetic", "operator", local_name)
        }

        pub fn apply_by_name(
            symbol: &str,
            arguments: &[Constant],
        ) -> Result<Constant, interpreter::Error> {
            let op = Operator::resolve(symbol)
                .ok_or_else(|| interpreter::Error::UnresolvedSymbol(qualified_name(symbol)))?;

            apply_operator(op, arguments)
        }

        fn apply_operator(
            op: Operator,
            arguments: &[Constant],
        ) -> Result<Constant, interpreter::Error> {
            if let &[lhs, rhs] = &arguments {
                self::apply(lhs, &op, rhs).ok_or_else(|| {
                    Error::ExpectedArguments(op.select().name().clone(), op.parameters())
                })
            } else {
                Err(interpreter::Error::ExpectedArguments(
                    op.select().name().clone(),
                    op.parameters(),
                ))
            }
        }

        fn apply(lhs: &Constant, op: &Operator, rhs: &Constant) -> Option<Constant> {
            match (lhs, op, rhs) {
                (Float(lhs), Plus, Float(rhs)) => Some(Float(lhs + rhs)),
                (Int(lhs), Plus, Int(rhs)) => Some(Int(lhs + rhs)),

                (Float(lhs), Minus, Float(rhs)) => Some(Float(lhs - rhs)),
                (Int(lhs), Minus, Int(rhs)) => Some(Int(lhs - rhs)),

                (Float(lhs), Times, Float(rhs)) => Some(Float(lhs * rhs)),
                (Int(lhs), Times, Int(rhs)) => Some(Int(lhs * rhs)),

                (Float(lhs), Divides, Float(rhs)) => Some(Float(lhs / rhs)),
                (Int(lhs), Divides, Int(rhs)) => Some(Int(lhs / rhs)),

                (Float(lhs), Modulo, Float(rhs)) => Some(Float(lhs % rhs)),
                (Int(lhs), Modulo, Int(rhs)) => Some(Int(lhs % rhs)),

                _otherwise => None,
            }
        }

        fn make_intrinsic_function(op: &Operator) -> ast::IntrinsicFunctionDef {
            ast::IntrinsicFunctionDef::new(op.select().name(), &op.parameters(), op.return_type())
        }

        pub fn declarations() -> interpreter::SymbolTable {
            [Plus, Minus, Times, Divides, Modulo]
                .iter()
                .map(|x| make_intrinsic_function(x))
                .map(|x| (x.name.clone(), ast::Declaration::IntrinsicFunction(x)))
                .collect()
        }
    }
}

mod text {
    use crate::{ast, runtime::interpreter};

    static PREFIX: &[&str] = &["builtins", "text"];

    pub fn in_prefix(name: &ast::Name) -> bool {
        PREFIX == name.scope_path
    }

    pub fn qualified_name(local_name: &str) -> ast::Name {
        ast::Name::intrinsic("arithmetic", "operator", local_name)
    }

    pub fn _apply_by_name(
        _symbol: &str,
        _arguments: &[ast::Constant],
    ) -> Result<ast::Constant, interpreter::Error> {
        todo!()
    }

    fn _interpolate(_components: Vec<ast::Constant>) -> ast::Constant {
        todo!()
    }
}

pub mod io {
    use crate::{ast, runtime::interpreter};

    static PREFIX: &[&str] = &["std", "io"];

    pub fn in_prefix(name: &ast::Name) -> bool {
        PREFIX == name.scope_path
    }

    pub fn qualified_name(local_name: &str) -> ast::Name {
        ast::Name::std("io", local_name)
    }

    pub fn apply_by_name(
        symbol: &str,
        arguments: &[ast::Constant],
    ) -> Result<ast::Constant, interpreter::Error> {
        match symbol {
            "print_line" if arguments.len() == 1 => {
                print_line(arguments[0].clone());
                Ok(ast::Constant::Void)
            }
            otherwise => Err(interpreter::Error::UnresolvedSymbol(qualified_name(
                otherwise,
            ))),
        }
    }

    pub fn declarations() -> interpreter::SymbolTable {
        // Perhaps this can be parsed later on
        // "fn print_line(line: <What here>) -> unit" => ast::InstrinsicFunctionDef
        let functions = vec![ast::IntrinsicFunctionDef::new(
            &qualified_name("print_line"),
            &[ast::Parameter::new(
                "line",
                ast::Type::simple(&ast::Name::simple("string")),
            )],
            ast::Type::Unit,
        )];

        functions
            .into_iter()
            .map(|x| (x.name().clone(), ast::Declaration::IntrinsicFunction(x)))
            .collect()
    }

    fn print_line(line: ast::Constant) {
        println!("{line}")
    }
}
