# Syntax and Grammar

This document outlines the syntax and grammar of the language, based on the parser implementation in `src/parser.rs` and the Abstract Syntax Tree (AST) defined in `src/ast.rs`.

## Program Structure

A program is a collection of function declarations. Only function declarations are allowed at the top level of a file.

```
// A program is one or more function declarations
fn main() -> i32 {
    0
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

## Comments

The language supports two types of comments:

-   **Line comments:** Start with `//` and continue to the end of the line.
-   **Block comments:** Start with `/*` and end with `*/`. Block comments can span multiple lines.

```
// This is a line comment.
fn main() -> i32 {
    /*
     * This is a multi-line
     * block comment.
     */
    let x: i32 = 10; // Comment at the end of a line
    x // implicit return
}
```

## Data Types

The language supports the following data types:

### Primitive Types

-   `i32`: 32-bit signed integer
-   `i64`: 64-bit signed integer
-   `f32`: 32-bit floating-point number
-   `f64`: 64-bit floating-point number
-   `bool`: Boolean (`true` or `false`)
-   `void`: Represents the absence of a value
-   `string`: String type (the type exists, but string literals are not yet supported).

### Function Types

Function types are defined using the `fn` keyword, followed by the parameter types and the return type.

```
// A function that takes two i32 parameters and returns an i32
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// A function with no parameters that returns void
fn do_nothing() -> void {
    // No return value, the function body is empty
}
```

## Variables

Variables can be declared as immutable (`let`) or mutable (`var`).

### Immutable Variables (`let`)

Immutable variables are declared using the `let` keyword. They must be initialized at declaration, and once assigned, their value cannot be changed.

```
let x: i32 = 10;
let y = 20; // Type can be inferred from the value
```

### Mutable Variables (`var`)

Mutable variables are declared using the `var` keyword. Their value can be changed after declaration. Initialization is optional.

```
var x: i32 = 10;
x = 20; // Allowed

var y: i32; // Allowed, value is uninitialized
y = 30;
```

## Statements

Statements are the building blocks of a function's body.

### Variable Declaration

```
let x: i32 = 10;
var y: i32 = 20;
var z: i32;
```

### Assignment

```
y = 30;
```

### Expression Statement

An expression can be used as a statement by appending a semicolon. The value of the expression is discarded.

```
add(10, 20);
```

## Expressions

Expressions produce a value.

### Literals

-   **Integer literals:** `10`, `42`
-   **Boolean literals:** `true`, `false`

### Binary Operations

The following binary operators are supported, with precedence rules similar to C++:

| Precedence | Operator | Associativity |
| :--- | :--- | :--- |
| 1 | `*`, `/` | Left-to-right |
| 2 | `+`, `-` | Left-to-right |
| 3 | `<`, `<=`, `>`, `>=` | Left-to-right |
| 4 | `==`, `!=` | Left-to-right |
| 5 | `&&` | Left-to-right |
| 6 | `||` | Left-to-right |

```
let x = 10 * 2 + 5;
let y = x > 20 && x < 30;
```

### Unary Operations

-   `-`: Negation
-   `!`: Logical NOT

```
let x = -10;
let y = !true;
```

### Function Calls

Functions are called using their name followed by parentheses containing the arguments.

```
add(10, 20);
```

### Variable References

A variable name in an expression evaluates to its value.

```
let x = 10;
let y = x;
```

### If Expression

The `if` construct is an expression that allows for conditional evaluation. It must have an `else` block, and both blocks must evaluate to a value of the same type.

```
let value = if x > 10 {
    1 // then_branch
} else {
    2 // else_branch
};
```

Chains of `else if` can be created by using another `if` expression in the `else` block.

```
let value = if x > 10 {
    1
} else {
    if x < 5 {
        2
    } else {
        3
    }
};
```

## Functions

Functions are defined using the `fn` keyword. They have a name, a list of parameters, a return type, and a body.

The `return` keyword is not supported. A function implicitly returns the value of the last expression in its body. If the last statement in the body is an expression without a trailing semicolon, its value is returned.

```
fn function_name(param1: type1, param2: type2) -> return_type {
    // function body
    value // implicit return
}

fn add(a: i32, b: i32) -> i32 {
    a + b // Implicitly returns the result of a + b
}

fn do_stuff() -> i32 {
    let x = 10;
    let y = 20;
    x + y; // This value is discarded
    x - y  // This value is returned
}
```