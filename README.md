# Hello MLIR

This project is a toy compiler for a simple, statically-typed programming language, built with Rust. It uses MLIR (Multi-Level Intermediate Representation) for code generation, targeting the x64 architecture.

## Features

### Compiler Features

- **Lexical Analysis**: Uses `logos` for efficient tokenization.
- **Parsing**: Implements a robust parser using the `chumsky` parser-combinator library.
- **Type Checking**: Features a type checker that supports scope-aware variable and function resolution.
- **Code Generation**: Generates MLIR using the `melior` crate and compiles it down to an executable.
- **Error Reporting**: Provides clear, formatted error messages using `ariadne`.
- **Testing**: Employs extensive snapshot testing with `insta`.

### Language Features

- **Static Typing**: All variables and functions have explicit or inferred types.
- **Functions**: Support for function declarations with parameters and return types.
- **Variables**: Immutable (`let`) and mutable (`var`) variable declarations.
- **Control Flow**: `if`/`else` expressions for conditional logic.
- **Expressions**: Rich support for binary and unary operations with correct precedence.
- **Types**: `i32`, `i64`, `f32`, `f64`, `bool`, and `void`.
- **Implicit Returns**: The last expression in a function body is automatically returned.

For a detailed guide, please see the [Syntax and Grammar Documentation](./docs/syntax.md).

## Prerequisites

- [Rust Toolchain](https://www.rust-lang.org/tools/install)
- [mise](https://mise.jdx.dev/): For managing project-specific tools and tasks.

## Getting Started

1.  **Install project-specific tools**:
    This command will install the correct Rust version and other tools defined in `.mise.toml`.
    ```sh
    mise install
    ```

2.  **Build the compiler**:
    ```sh
    mise run build
    ```
    The executable will be available at `target/release/hello-mlir`.

## Usage

The compiler can be run directly using `cargo run`. It accepts a source file and can be run in two modes:

1.  **AST Output (`--mode ast`)**: Parses the code, type-checks it, and prints the Abstract Syntax Tree (AST) as YAML to standard output.

    ```sh
    cargo run -- samples/main.txt --mode ast
    ```

2.  **Machine Code Generation (`--mode machine-code`)**: Compiles the code down to a native executable. Use the `--output` flag to specify the name of the executable file.

    ```sh
    # Compile the source file and create an executable named 'main'
    cargo run -- samples/main.txt --mode machine-code --output main

    # Run the generated executable
    ./main

    # Check the exit code
    echo $?
    ```

    If the `--output` flag is omitted, the generated MLIR code will be printed to standard output.

## Testing

Run the test suite using `mise`:

```sh
# Run all tests
mise run test
```

This project uses `insta` for snapshot testing. If you make a change that affects a snapshot, you can review and approve the changes interactively:

```sh
# Review and update snapshots
mise run test-review
```

## Project Structure

- `src/main.rs`: Command-line interface and main entry point.
- `src/token.rs`: Lexer definitions using `logos`.
- `src/ast.rs`: Abstract Syntax Tree (AST) node definitions.
- `src/parser.rs`: Parser logic using `chumsky`.
- `src/typechecker.rs`: Type checking and semantic analysis.
- `src/codegen.rs`: MLIR code generation using `melior`.
- `samples/`: Example source code files for the language.
- `docs/`: Project documentation.
