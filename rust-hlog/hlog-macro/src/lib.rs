use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse::Parser, parse_macro_input};

#[proc_macro_attribute]
pub fn hlog(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as DeriveInput);
    match &mut ast.data {
        syn::Data::Struct(struct_data) => {
            if let syn::Fields::Named(fields) = &mut struct_data.fields {
                fields.named.push(
                    syn::Field::parse_named
                        .parse2(quote! { pub hlog: ::rust_hlog::HLog })
                        .unwrap(),
                );
            }

            quote! {
                #ast
            }
            .into()
        }
        _ => panic!("`add_field` has to be used with structs "),
    }
}
