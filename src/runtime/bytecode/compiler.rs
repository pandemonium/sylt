use std::rc;

use crate::ast;

use super::{model, vm, builtins};

pub fn make_executable(program: ast::Program) -> vm::Executable {
    let mut compile = Compile::default();
    builtins::provide_standard_library(&mut compile);

    compile.program(program);
    compile.link()
}

#[derive(Default)]
pub struct Compile(Generator);

impl Compile {
    fn link(self) -> vm::Executable {
        self.0.into_executable()
    }

    pub fn register_builtin(&mut self, builtin: model::BuiltinFunction) {
        self.0.builtins.push(builtin)
    }

    fn program(&mut self, program: ast::Program) {
        for decl in program.declarations {
            self.declaration(decl)
        }

        let entry_point = self.make_label();
        self.select_target(entry_point);

        self.block(program.entry_point);

        if !self.0.current_block().is_terminated() {
            self.emit_unit_return()
        }
    }

    fn declaration(&mut self, declaration: ast::Declaration) {
        match declaration {
            ast::Declaration::Function(declarator) => {
                let target = self.0.emit_function(&declarator.name);
                self.select_target(target);

                for arg in declarator.parameters.into_iter() {
                    // this is probably where a type check needs to happen
                    // Can StoreLocal be made aware of types?
                    // Or just something like: Bytecode::TypeCheck(ast::Type) to verify that
                    // the top of the stack is what I expect.
                    let slot = self.0.allocate_local_slot(arg.name);
                    self.0.emit(model::Bytecode::StoreLocal(slot as u8))
                }

                self.block(declarator.body);

                println!("Current block: {}", self.0.current_block_id);

                if !self.0.current_block().is_terminated() {
                    self.emit_unit_return()
                }
            }
            _otherwise => todo!(),
        }
    }

    fn emit_unit_return(&mut self) {
        println!("Adding a return to block {}", self.0.current_block_id);
        self.0.emit(model::Bytecode::LoadConstant(model::Value::Unit));
        self.0.emit(model::Bytecode::Return)
    }

    fn block(&mut self, block: ast::Block) {
        for stmt in block.statements {
            self.statement(stmt)
        }
    }

    fn make_label(&mut self) -> model::Label {
        model::Label(self.0.make_block())
    }

    fn select_target(&mut self, model::Label(block_id): model::Label) {
        self.0.unsafe_set_current_target(block_id as usize)
    }

    fn statement(&mut self, statement: ast::Statement) {
        match statement {
            ast::Statement::Let { lhs, rhs } => {
                let name = ast::Name::simple(&lhs);
                let index = if let Some(slot) = self.0.resolve_local_slot(&name) {
                    slot
                } else {
                    self.0.allocate_local_slot(name)
                };
                self.expression(rhs);
                self.0.emit(model::Bytecode::StoreLocal(index as u8))
            }
            ast::Statement::If {
                predicate,
                consequent,
                alternate,
            } => {
                let consequent_target = self.make_label();
                let alternate_taret = self.make_label();

                // This creates an orphan block when either or both if blocks ends in a return.
                let end_target = self.make_label();

                self.expression(predicate);
                self.0.emit(model::Bytecode::ConditionalJump(
                    consequent_target,
                    alternate_taret,
                ));

                self.select_target(consequent_target);
                self.block(consequent);
                if !self.0.current_block().is_terminated() {
                    self.0.emit(model::Bytecode::Jump(end_target));
                }

                self.select_target(alternate_taret);
                self.block(alternate);
                if !self.0.current_block().is_terminated() {
                    self.0.emit(model::Bytecode::Jump(end_target));
                }

                self.select_target(end_target)
                // The when_true and when_false blocks both need this block
                // identifiers as its parent scope, somehow.
            }
            ast::Statement::While { predicate, body } => {
                let eval_predicate = self.make_label();
                let body_target = self.make_label();
                let end_target = self.make_label();

                self.0.emit(model::Bytecode::Jump(eval_predicate));
                self.select_target(eval_predicate);

                self.expression(predicate);
                self.0
                    .emit(model::Bytecode::ConditionalJump(body_target, end_target));

                self.select_target(body_target);
                self.block(body);
                self.0.emit(model::Bytecode::Jump(eval_predicate));

                self.select_target(end_target);
            }
            ast::Statement::Expression(expression) => {
                self.expression(expression);
                self.0.emit(model::Bytecode::Discard)
            }
            ast::Statement::Return(expression) => {
                self.expression(expression);
                self.0.emit(model::Bytecode::Return)
            }
        }
    }

    fn expression(&mut self, expression: ast::Expression) {
        let Compile(gen) = self;
        match expression {
            ast::Expression::Literal(constant) => gen.emit(model::Bytecode::LoadConstant(constant.into())),
            ast::Expression::Variable(name) => {
                // This is getting to be a pain. What value does this add?
                let slot = gen
                    .resolve_local_slot(&name.as_value().expect("type selector used in variable"))
                    .expect(&format!("Unresolved symbol {name:?}"))
                    as u8;
                gen.emit(model::Bytecode::LoadLocal(slot))
            }
            ast::Expression::ApplyInfix { lhs, symbol, rhs } => {
                self.expression(*lhs);
                self.expression(*rhs);
                self.operator(symbol)
            }
            ast::Expression::Apply { symbol, arguments } => {
                // Where does it validate formals versus actuals?
                for arg in arguments.into_iter().rev() {
                    self.expression(arg)
                }

                if let Some(index) = self.0.lookup_function_index(&symbol) {
                    self.0.emit(model::Bytecode::Invoke(index as u16))
                } else if let Some(index) = self.0.lookup_builtin_index(&symbol) {
                    self.0.emit(model::Bytecode::InvokeBuiltin(index as u16))
                } else {
                    panic!("Undefined symbol: {symbol:?}")
                }
            }
        }
    }

    fn operator(&mut self, operator: ast::Operator) {
        let bytecode = match operator {
            ast::Operator::Plus => model::Bytecode::Arithmetic(model::AluOp::Add),
            ast::Operator::Minus => model::Bytecode::Arithmetic(model::AluOp::Subtract),
            ast::Operator::Times => model::Bytecode::Arithmetic(model::AluOp::Multiply),
            ast::Operator::Divides => model::Bytecode::Arithmetic(model::AluOp::Divide),
            ast::Operator::Modulo => model::Bytecode::Arithmetic(model::AluOp::Modulo),
            ast::Operator::LT => model::Bytecode::Arithmetic(model::AluOp::Lt),
            ast::Operator::LTE => model::Bytecode::Arithmetic(model::AluOp::Lte),
            ast::Operator::GT => model::Bytecode::Arithmetic(model::AluOp::Gt),
            ast::Operator::GTE => model::Bytecode::Arithmetic(model::AluOp::Gte),
            ast::Operator::Equals => model::Bytecode::Arithmetic(model::AluOp::Equals),
            ast::Operator::NotEqual => model::Bytecode::Arithmetic(model::AluOp::NotEqual),
            ast::Operator::And => model::Bytecode::Arithmetic(model::AluOp::And),
            ast::Operator::Or => model::Bytecode::Arithmetic(model::AluOp::Or),
        };

        self.0.emit(bytecode)
    }
}

#[derive(Default)]
struct Generator {
    identifiers: Vec<String>,
    constants: Vec<ast::Constant>,
    functions: Vec<(ast::Name, model::Label)>,
    current_block_id: usize,
    blocks: Vec<model::BasicBlock>,
    locals: Vec<ast::Name>,
    builtins: Vec<model::BuiltinFunction>,
}

impl Generator {
    fn into_executable(self) -> vm::Executable {
        vm::Executable {
            functions: self
                .functions
                .iter()
                .map(|(name, target)| model::Function {
                    name: name.clone(),
                    target: *target,
                })
                .collect(),
            builtins: self.builtins,
            constants: self.constants.into_iter().map(Into::into).collect(),
            entry_point_block_id: self.current_block_id,
            blocks: self.blocks,
        }
    }

    fn emit_function(&mut self, name: &ast::Name) -> model::Label {
        let label = model::Label(self.make_block());
        self.functions.push((name.clone(), label));

        label
    }

    fn lookup_function_index(&self, name: &ast::Select) -> Option<usize> {
        name.as_value()
            .and_then(|name| self.functions.iter().position(|(nm, _)| nm == name))
    }

    fn lookup_builtin_index(&self, name: &ast::Select) -> Option<usize> {
        name.as_value().and_then(|name| {
            self.builtins
                .iter()
                .position(|builtin| &builtin.name == name)
        })
    }

    fn make_block(&mut self) -> u16 {
        self.blocks.push(model::BasicBlock::default());
        (self.blocks.len() - 1) as u16
    }

    fn intern_constant(&mut self, c: ast::Constant) -> usize {
        if let Some(index) = self.constants.iter().position(|x| *x == c) {
            index
        } else {
            self.constants.push(c);
            self.constants.len() - 1
        }
    }

    fn intern_identifier(&mut self, id: String) -> usize {
        if let Some(index) = self.identifiers.iter().position(|x| x == &id) {
            index
        } else {
            self.identifiers.push(id);
            self.identifiers.len() - 1
        }
    }

    fn emit(&mut self, code: model::Bytecode) {
        self.blocks[self.current_block_id].instructions.push(code);
    }

    fn unsafe_set_current_target(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            self.current_block_id = block_id
        } else {
            panic!("Block id out of bounds")
        }
    }

    fn current_block(&self) -> &model::BasicBlock {
        &self.blocks[self.current_block_id]
    }

    fn lookup_block(&self, model::Label(block_id): &model::Label) -> &model::BasicBlock {
        &self.blocks[*block_id as usize]
    }

    fn allocate_local_slot(&mut self, symbol: ast::Name) -> usize {
        self.locals.push(symbol);
        self.locals.len() - 1
    }

    fn resolve_local_slot(&self, symbol: &ast::Name) -> Option<usize> {
        self.locals.iter().position(|x| x == symbol)
    }
}