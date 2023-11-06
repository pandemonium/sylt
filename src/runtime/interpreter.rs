use std::collections;
use crate::{ast, runtime::intrinsics};

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
    ExpectedReturn(ast::Type),
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
    locals: collections::HashMap<String, ast::Constant>,
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
    ) -> Result<&ast::IntrinsicFunctionDef, Error> {
        if let Some(ast::Declaration::IntrinsicFunction(def)) = self.find_function(symbol) {
            Ok(def)
        } else {
            Err(Error::UnresolvedSymbol(symbol.clone()))
        }
    }

    //    fn resolve_

    fn get_local_variable(&self, symbol: &ast::Name) -> Option<&ast::Constant> {
        self.locals.get(&symbol.name)
    }

    fn put_local_variable(&mut self, symbol: &ast::Name, value: ast::Constant) {
        self.locals.insert(symbol.name.clone(), value);
    }

    fn set_return_value(&mut self, return_value: ast::Constant) {
        self.return_value = Some(return_value)
    }

    fn verify_return_value(&self, of_type: &ast::Type) -> Result<ast::Constant, Error> {
        if of_type != &ast::Type::Unit {
            self.return_value
                .as_ref()
                .filter(|value| &value.get_type() == of_type)
                .ok_or_else(|| Error::ExpectedReturn(of_type.clone()))
                .cloned()
        } else {
            Ok(ast::Constant::Void)
        }
    }

    fn reduce(&self, e: &ast::Expression) -> Result<ast::Constant, Error> {
        match e {
            ast::Expression::Literal(constant) => Ok(constant.clone()),
            ast::Expression::Variable(symbol) => self
                .get_local_variable(symbol)
                .cloned()
                .ok_or_else(|| Error::UnresolvedSymbol(symbol.clone())),
            ast::Expression::ApplyInfix { lhs, symbol, rhs } => {
                let lhs = self.reduce(&lhs)?;
                let rhs = self.reduce(&rhs)?;
                self.reduce(&ast::Expression::Apply {
                    symbol: symbol.select(),
                    arguments: vec![ast::Expression::Literal(lhs), ast::Expression::Literal(rhs)],
                })
            }
            ast::Expression::Apply { symbol, arguments } => {
                self.apply_symbol(e, symbol, &arguments)
            }
        }
    }

    fn apply_symbol(
        &self,
        apply: &ast::Expression,
        symbol: &ast::Select,
        arguments: &[ast::Expression],
    ) -> Result<ast::Constant, Error> {
        match symbol {
            ast::Select::Function(name) => self.apply_function(name, arguments),
            ast::Select::Intrinsic(name) => self.apply_intrinsic(name, arguments),
            ast::Select::Type(name) => Err(Error::ExpectedFunction(name.clone())),
        }
    }

    fn apply_intrinsic(
        &self,
        symbol: &ast::Name,
        arguments: &[ast::Expression],
    ) -> Result<ast::Constant, Error> {
        let target = self.resolve_intrinsic_function(symbol)?;
        let arguments = arguments
            .iter()
            // This has to be the same number.
            .take(target.parameters.len())
            .map(|p| self.reduce(p))
            .collect::<Result<Vec<_>, Error>>()?;

        intrinsics::invoke_intrinsic_function(target, &arguments)
    }

    fn apply_function(
        &self,
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
                // Verify that the type of `value` matches parameter.type
                let value = self.reduce(argument)?;
                let name = ast::Name::simple(&parameter.name);
                frame.put_local_variable(&name, value);
            }
            frame.interpret_block(&function.body)?;
            frame.verify_return_value(&function.return_type)
        } else {
            Err(Error::ExpectedArguments(
                symbol.clone(),
                function.parameters.clone(),
            ))
        }
    }

    // This could actually consume self. Right?
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
                    when_true,
                    when_false,
                } => {
                    if self.evaluate_boolean(predicate)? {
                        self.interpret_block(when_true)?;
                    } else {
                        self.interpret_block(when_false)?;
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

pub struct Interpreter {
    environment: Environment,
    program: ast::Program,
}

impl Interpreter {
    pub fn new(program: ast::Program, intrinsics: SymbolTable) -> Self {
        let program_definitions = program.definitions.iter().fold(
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
        frame.verify_return_value(&ast::Constant::Int(0).get_type())
    }
}
