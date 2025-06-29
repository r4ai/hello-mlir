# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

This project uses `mise` for task management. The following commands are available:

- **Build**: `mise run build`
- **Test**: `mise run test`
- **Review test snapshots**: `mise run test-review`
- **Format code**: `mise run format-write`
- **Check formatting**: `mise run format`
- **Lint**: `mise run lint`
- **Fix linting issues**: `mise run lint-write`

## Architecture Overview

This is a compiler project for a simple integer-focused language that targets MLIR (Multi-Level Intermediate Representation). The compiler is written in Rust and follows a traditional compiler pipeline:

### Core Components

1. **Lexical Analysis** (`src/token.rs`): Tokenizes input using the `logos` crate
2. **Parsing** (`src/parser.rs`): Builds an AST using the `chumsky` parser combinator library
3. **AST Definition** (`src/ast.rs`): Defines the complete AST structure with spans for error reporting
4. **Type Checking** (`src/typechecker.rs`): Performs semantic analysis and type checking
5. **Code Generation** (`src/codegen.rs`): Targets MLIR using the `melior` crate (currently incomplete)

### Key Features

- **Comprehensive AST**: Supports expressions (literals, binary/unary ops, function calls, variables, if expressions), statements (function declarations, variable declarations, assignments), and a full type system.
- **Error Reporting**: Uses `ariadne` for beautiful error messages with source spans.
- **Snapshot Testing**: Extensive test coverage using `insta` for snapshot testing.
- **Two Compilation Modes**:
  - `--mode ast`: Outputs AST as YAML.
  - `--mode machine-code`: Full compilation (type checking + MLIR generation).

### Dependencies

- **Parser**: `chumsky` for parser combinators, `logos` for lexing.
- **MLIR**: `melior` for MLIR code generation.
- **Error Handling**: `ariadne` for diagnostics, `anyhow` and `thiserror` for error handling.
- **CLI**: `clap` for command-line interface.
- **Serialization**: `serde` and `serde_yaml` for AST output.
- **Testing**: `insta` for snapshot tests, `pretty_assertions` for better test output.

### Testing Strategy

The project uses snapshot testing extensively with `insta`. Test snapshots are stored in `src/snapshots/` and cover:

- Parser functionality for all language constructs.
- Type checker behavior for various scenarios.
- Error cases and recovery.

When making changes to parser or type checker output, always run `mise run test-review` to review and approve snapshot changes.

### Language Features

The language supports:

- Integer and boolean literals.
- Binary operations (arithmetic, comparison, logical).
- Unary operations (negation, logical not).
- Variable declarations (immutable `let` and mutable `var`).
- Function declarations with parameters and return types.
- Function calls.
- If/else expressions.
- Assignment statements.
- Type annotations (i32, i64, f32, f64, bool, void, string, function types).
- Implicit returns (the last expression in a function body is the return value).

### Development Notes

- Machine code generation is incomplete (marked with `todo!()`)
- The project uses Rust 2024 edition
- Development tools are managed via `mise` (successor to `asdf`)
- Git hooks are configured via `lefthook` but currently commented out
- The compiler CLI accepts input files and can output either AST or compiled machine code
