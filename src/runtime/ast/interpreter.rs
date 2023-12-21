use crate::ast;
use std::collections;

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    UnresolvedSymbol(ast::Name),
    ExpectedFunction(ast::Name),
    ExpectedArguments(ast::Name, Vec<ast::Parameter>),
    ExpectedType {
        expected: ast::Name,
        was: ast::Name,
        at: ast::Expression,
    },
    ExpectedReturn(ast::Select),
}

enum Environment {
    Builtins(SymbolTable),
    Extend(SymbolTable, Box<Environment>),
}

pub type SymbolTable = collections::HashMap<ast::Name, ast::Declaration>;

impl Environment {
    fn find_function(&self, symbol: &ast::Name) -> Option<&ast::Declaration> {
        match self {
            Environment::Builtins(root) => root.get(&symbol),
            Environment::Extend(definitions, parent) => definitions
                .get(&symbol)
                .or_else(|| parent.find_function(symbol)),
        }
    }
}

// Does this hold the return value?
// fn set_return_value()?
struct ActivationFrame<'a> {
    interpreter: &'a Interpreter,

    // Perhaps this should be a hierarchical thing for lexical scopes
    locals: collections::HashMap<ast::Name, ast::Constant>,
    return_value: Option<ast::Constant>,
}

impl<'a> ActivationFrame<'a> {
    fn new(interpreter: &'a Interpreter) -> Self {
        Self {
            interpreter,
            locals: Default::default(),
            return_value: None,
        }
    }

    fn find_function(&self, symbol: &ast::Name) -> Option<&ast::Declaration> {
        self.interpreter.environment.find_function(symbol)
    }

    fn resolve_intrinsic_function(
        &self,
        symbol: &ast::Name,
    ) -> Result<&ast::IntrinsicFunctionDeclarator, Error> {
        if let Some(ast::Declaration::IntrinsicFunction(def)) = self.find_function(symbol) {
            Ok(def)
        } else {
            Err(Error::UnresolvedSymbol(symbol.clone()))
        }
    }

    fn get_local_variable(&self, symbol: &ast::Select) -> Option<&ast::Constant> {
        self.locals.get(symbol.as_value().expect("Select::Value"))
    }

    fn put_local_variable(&mut self, symbol: &ast::Name, value: ast::Constant) {
        self.locals.insert(symbol.clone(), value);
    }

    fn set_return_value(&mut self, return_value: ast::Constant) {
        self.return_value = Some(return_value)
    }

    fn types_unify(lhs: &ast::Type, rhs: &ast::Type) -> bool {
        let res = lhs.subsumes(rhs);

        if !res {
            println!("{lhs} and {rhs} do not unify");
        }

        res
    }

    fn typecheck_return(&self, of_type: &ast::Select) -> Result<ast::Constant, Error> {
        self.return_value
            .as_ref()
            .filter(|value| of_type.subsumes(&value.get_type()))
            .ok_or_else(|| Error::ExpectedReturn(of_type.clone()))
            .cloned()
    }

    fn apply_intrinsic(
        &self,
        _apply: &ast::Expression,
        symbol: &ast::Name,
        arguments: &[ast::Expression],
    ) -> Result<ast::Constant, Error> {
        let intrinsic = self.resolve_intrinsic_function(symbol)?;
        let arguments = arguments
            .iter()
            .take(intrinsic.parameters.len())
            .map(|p| self.reduce(p))
            .collect::<Result<Vec<_>, Error>>()?;

        intrinsic.dispatch.invoke(&arguments)
    }

    fn apply_function(
        &self,
        apply: &ast::Expression,
        symbol: &ast::Name,
        arguments: &[ast::Expression],
    ) -> Result<ast::Constant, Error> {
        let target = self
            .find_function(symbol)
            .ok_or_else(|| Error::UnresolvedSymbol(symbol.clone()))?;

        let function = target
            .as_function()
            .ok_or_else(|| Error::ExpectedFunction(symbol.clone()))?;

        if arguments.len() == function.parameters.len() {
            let mut frame = Self::new(self.interpreter);
            for (argument, parameter) in arguments.iter().zip(&function.parameters) {
                let name = ast::Name::simple(parameter.get_name());
                let value = self.reduce(argument)?;
                if parameter.get_type().subsumes(&value.get_type()) {
                    frame.put_local_variable(&name, value);
                } else {
                    Err(Error::ExpectedType {
                        expected: parameter.get_type().name().clone(),
                        was: value.get_type().name(),
                        at: apply.clone(),
                    })?;
                }
            }
            frame.interpret_block(&function.body)?;
            frame.typecheck_return(&function.return_type)
        } else {
            Err(Error::ExpectedArguments(
                symbol.clone(),
                function.parameters.clone(),
            ))
        }
    }

    fn reduce(&self, e: &ast::Expression) -> Result<ast::Constant, Error> {
        match e {
            ast::Expression::Literal(constant) => Ok(constant.clone()),
            ast::Expression::Variable(symbol) => self
                .get_local_variable(symbol)
                .cloned()
                .ok_or_else(|| Error::UnresolvedSymbol(symbol.name().clone())),
            ast::Expression::ApplyInfix { lhs, symbol, rhs } => {
                let lhs = self.reduce(&lhs)?;
                let rhs = self.reduce(&rhs)?;
                self.apply_intrinsic(
                    e,
                    &symbol.name(),
                    &[ast::Expression::Literal(lhs), ast::Expression::Literal(rhs)],
                )
            }
            ast::Expression::Apply { symbol, arguments } => {
                self.apply_function(e, symbol.as_value().expect("Select::Value"), &arguments)
            }
        }
    }

    fn interpret_block(&mut self, block: &ast::Block) -> Result<(), Error> {
        for statement in &block.statements {
            match statement {
                ast::Statement::Let { lhs, rhs } => {
                    let lhs = ast::Name::simple(&lhs);
                    let rhs = self.reduce(rhs)?;
                    self.put_local_variable(&lhs, rhs);
                }
                ast::Statement::If {
                    predicate,
                    consequent,
                    alternate,
                } => {
                    if self.evaluate_boolean(predicate)? {
                        self.interpret_block(consequent)?;
                    } else {
                        self.interpret_block(alternate)?;
                    }
                }
                ast::Statement::While { predicate, body } => {
                    while self.evaluate_boolean(predicate)? {
                        self.interpret_block(body)?;
                    }
                }
                ast::Statement::Expression(e) => {
                    self.reduce(e)?;
                }
                ast::Statement::Return(return_value) => {
                    let return_value = self.reduce(return_value)?;
                    self.set_return_value(return_value);
                    break;
                }
            }
        }

        Ok(())
    }

    fn evaluate_boolean(&self, e: &ast::Expression) -> Result<bool, Error> {
        let value = self.reduce(e)?;
        let value = value.as_boolean().ok_or_else(|| Error::ExpectedType {
            expected: ast::Constant::Boolean(true).get_type().name().clone(),
            was: value.get_type().name().clone(),
            at: e.clone(),
        })?;

        Ok(*value)
    }
}

// This isn't the right composition
pub struct Interpreter {
    environment: Environment,
    program: ast::Program,
}

impl Interpreter {
    pub fn new(program: ast::Program, intrinsics: SymbolTable) -> Self {
        let program_definitions = program.declarations.iter().fold(
            collections::HashMap::<_, ast::Declaration>::new(),
            |mut table, def| {
                table.insert(def.name().clone(), def.clone());
                table
            },
        );

        Self {
            environment: Environment::Extend(
                program_definitions,
                Box::new(Environment::Builtins(intrinsics)),
            ),
            program,
        }
    }

    // This could return some kind of AbstractIoProgram
    pub fn run(&self) -> Result<ast::Constant, Error> {
        let mut frame = ActivationFrame::new(self);
        frame.interpret_block(&self.program.entry_point)?;
        frame.typecheck_return(&ast::Select::primitive_type(ast::PrimitiveType::Int))
    }
}

#[cfg(test)]
mod tests {
    use super::{ActivationFrame, Interpreter};
    use crate::{
        ast::{self, Constant, Program},
        runtime::ast::intrinsics,
        syntax, Error,
    };

    fn make_interpreter() -> Interpreter {
        Interpreter {
            environment: super::Environment::Builtins(intrinsics::initialize()),
            program: Program {
                declarations: vec![],
                entry_point: crate::ast::Block { statements: vec![] },
            },
        }
    }

    fn run_program(source: &str) -> Result<ast::Constant, Error> {
        let program = syntax::analyze(&source.chars().collect::<Vec<_>>())?;
        let interpreter = Interpreter::new(program, intrinsics::initialize());
        Ok(interpreter.run()?)
    }

    fn eval_expr(expr: &str, result: Constant) {
        let intp = make_interpreter();
        let frame = ActivationFrame::new(&intp);
        let expr = syntax::parse_expression(expr).unwrap();
        assert_eq!(frame.reduce(&expr).unwrap(), result);
    }

    #[test]
    fn operator_precedence() {
        eval_expr("1 < 2 and 3 + 4 >= 5", Constant::Boolean(true));
        eval_expr("1 + 2 * 3", Constant::Int(7));
        eval_expr("2 * 3 + 1", Constant::Int(7));
        eval_expr("2 * (3 + 1)", Constant::Int(8));
        eval_expr("(3 + 1) * 2", Constant::Int(8));

        eval_expr("1.0 + 2.0 * 3.0", Constant::Float(7.0));
        eval_expr("2.0 * 3.0 + 1.0", Constant::Float(7.0));
        eval_expr("2.0 * (3.0 + 1.0)", Constant::Float(8.0));
        eval_expr("(3.0 + 1.0) * 2.0", Constant::Float(8.0));

        eval_expr("1 < (3 + 1) * 2", Constant::Boolean(true));
        eval_expr("1 > (3 + 1) * 2", Constant::Boolean(false));
        eval_expr("1 <= (3 + 1) * 2", Constant::Boolean(true));
        eval_expr("1 >= (3 + 1) * 2", Constant::Boolean(false));

        eval_expr(
            "1 < (3 + 1) * 2 and 1 < (3 + 1) * 2",
            Constant::Boolean(true),
        );
        eval_expr(
            "1 < (3 + 1) * 2 or 1 < (3 + 1) * 2",
            Constant::Boolean(true),
        );
        eval_expr(
            "1 < (3 + 1) * 2 and 1 > (3 + 1) * 2",
            Constant::Boolean(false),
        );
        eval_expr(
            "1 > (3 + 1) * 2 or 1 > (3 + 1) * 2",
            Constant::Boolean(false),
        );
    }

    #[test]
    fn fibonacci() {
        let source = r#"
        fn fibonacci(n: Int) -> Int {
            if n == 0 {
                return 0;
            } else {
                if n == 1 {
                    return 1;
                } else {
                    return fibonacci(n - 1) + fibonacci(n - 2);
                }
            }
        }

        {
            return fibonacci(20);
        }
        "#;
        let result = run_program(source).unwrap();
        println!("{result:#?}");
    }
}
