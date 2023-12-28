use crate::ast;
use super::{builtins, model, vm};

pub fn make(program: ast::Program) -> vm::Executable {
    let mut compile = Compile::default();
    builtins::provide_standard_library(&mut compile);

    compile.program(program);
    compile.link()
}

#[derive(Default)]
pub struct Compile(Generator);

impl Compile {
    fn link(self) -> vm::Executable {
        self.0.build_executable()
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
                    self.emit(model::Bytecode::StoreLocal(slot as u8))
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
        self.emit(model::Bytecode::LoadConstant(model::Value::Unit));
        self.emit(model::Bytecode::Return)
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
                self.emit(model::Bytecode::StoreLocal(index as u8))
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
                self.emit(model::Bytecode::ConditionalJump(
                    consequent_target,
                    alternate_taret,
                ));

                self.select_target(consequent_target);
                self.block(consequent);
                if !self.0.current_block().is_terminated() {
                    self.emit(model::Bytecode::Jump(end_target));
                }

                self.select_target(alternate_taret);
                self.block(alternate);
                if !self.0.current_block().is_terminated() {
                    self.emit(model::Bytecode::Jump(end_target));
                }

                self.select_target(end_target)
                // The when_true and when_false blocks both need this block
                // identifiers as its parent scope, somehow.
            }
            ast::Statement::While { predicate, body } => {
                let eval_predicate = self.make_label();
                let loop_body = self.make_label();
                let end = self.make_label();

                self.emit(model::Bytecode::Jump(eval_predicate));
                self.select_target(eval_predicate);

                self.expression(predicate);
                self.emit(model::Bytecode::ConditionalJump(loop_body, end));

                self.select_target(loop_body);
                self.block(body);
                self.emit(model::Bytecode::Jump(eval_predicate));

                self.select_target(end);
            }
            ast::Statement::Expression(expression) => {
                self.expression(expression);
                self.emit(model::Bytecode::Discard)
            }
            ast::Statement::Return(expression) => {
                self.expression(expression);
                self.emit(model::Bytecode::Return)
            }
            ast::Statement::ArrayUpdate { array, subscript, rhs } => {
                todo!()
            }
        }
    }

    fn expression(&mut self, expression: ast::Expression) {
        let Compile(gen) = self;
        match expression {
            ast::Expression::Literal(constant) => {
                self.emit(model::Bytecode::LoadConstant(constant.into()))
            }
            ast::Expression::Variable(name) => {
                // This is getting to be a pain. What value does this add?
                let slot = gen
                    .resolve_local_slot(&name.as_value().expect("type selector used in variable"))
                    .expect(&format!("Unresolved symbol {name:?}"))
                    as u8;
                self.emit(model::Bytecode::LoadLocal(slot))
            }
            ast::Expression::GetArrayElement { array, subscript } => {
                todo!()
            }
            ast::Expression::PutArrayElement { array, subscript, element } => {
                todo!()
            }
            ast::Expression::ApplyInfix { lhs, symbol, rhs } => {
                self.expression(*lhs);
                self.expression(*rhs);
                self.operator(symbol)
            }
            ast::Expression::Apply { symbol, arguments } => {
                let arg_count = arguments.len();
                // Where does it validate formals versus actuals?
                for arg in arguments {
                    self.expression(arg)
                }

                if let Some(index) = self.0.lookup_function_index(&symbol) {
                    self.emit(model::Bytecode::Invoke(index as u16))
                } else if let Some(index) = self.0.lookup_builtin_index(&symbol) {
                    self.emit(model::Bytecode::InvokeBuiltin(
                        index as u16,
                        arg_count as u8,
                    ))
                } else {
                    panic!("Undefined symbol: {symbol:?}")
                }
            }
        }
    }

    fn operator(&mut self, operator: ast::Operator) {
        self.emit(model::Bytecode::Arithmetic(operator.into()))
    }

    fn emit(&mut self, code: model::Bytecode) {
        self.0.append_bytecode(code)
    }
}

#[derive(Default)]
struct Generator {
    constants: Vec<ast::Constant>,
    functions: Vec<(ast::Name, model::Label)>,
    current_block_id: usize,
    blocks: Vec<model::BasicBlock>,
    locals: Vec<ast::Name>,
    builtins: Vec<model::BuiltinFunction>,
}

impl Generator {
    fn build_executable(self) -> vm::Executable {
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

    fn append_bytecode(&mut self, code: model::Bytecode) {
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

    fn allocate_local_slot(&mut self, symbol: ast::Name) -> usize {
        self.locals.push(symbol);
        self.locals.len() - 1
    }

    fn resolve_local_slot(&self, symbol: &ast::Name) -> Option<usize> {
        self.locals.iter().position(|x| x == symbol)
    }
}
