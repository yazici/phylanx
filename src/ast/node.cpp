//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/ast/node.hpp>

#include <hpx/include/serialization.hpp>
#include <hpx/include/util.hpp>

#include <iosfwd>

namespace phylanx { namespace ast
{
    ///////////////////////////////////////////////////////////////////////////
    std::int64_t tagged::next_id = 0;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        static constexpr int const precedence[] =
        {
            // precedence 1
            1, // op_comma

            // precedence 2
            2, // op_assign
            2, // op_plus_assign
            2, // op_minus_assign
            2, // op_times_assign
            2, // op_divide_assign
            2, // op_mod_assign
            2, // op_bit_and_assign
            2, // op_bit_xor_assign
            2, // op_bitor_assign
            2, // op_shift_left_assign
            2, // op_shift_right_assign

            // precedence 3
            3, // op_logical_or

            // precedence 4
            4, // op_logical_and

            // precedence 5
            5, // op_bit_or

            // precedence 6
            6, // op_bit_xor

            // precedence 7
            7, // op_bit_and

            // precedence 8
            8, // op_equal
            8, // op_not_equal

            // precedence 9
            9, // op_less
            9, // op_less_equal
            9, // op_greater
            9, // op_greater_equal

            // precedence 10
            10, // op_shift_left
            10, // op_shift_right

            // precedence 11
            11, // op_plus
            11, // op_minus

            // precedence 12
            12, // op_times
            12, // op_divide
            12, // op_mod

            // precedence 13
            13, // op_positive
            13, // op_negative
            13, // op_pre_incr
            13, // op_pre_decr
            13, // op_compl
            13, // op_not

            // precedence 14
            14, // op_post_incr
            14  // op_post_decr
        };
    }

    int precedence_of(optoken t)
    {
        int op = static_cast<int>(t);
        HPX_ASSERT(op >= 0 && op < static_cast<int>(optoken::op_unknown) -1);
        return detail::precedence[op];
    }

    ///////////////////////////////////////////////////////////////////////////
    void serialize(hpx::serialization::input_archive& ar, optoken& id, unsigned)
    {
        int val;
        ar >> val;
        id = static_cast<optoken>(val);
    }

    void serialize(
        hpx::serialization::output_archive& ar, optoken const& id, unsigned)
    {
        int val = static_cast<int>(id);
        ar << val;
    }

    ///////////////////////////////////////////////////////////////////////////
    void identifier::serialize(hpx::serialization::output_archive& ar, unsigned)
    {
        ar << name << id;
    }

    void identifier::serialize(hpx::serialization::input_archive& ar, unsigned)
    {
        ar >> name >> id;
    }

    ///////////////////////////////////////////////////////////////////////////
    void primary_expr::serialize(
        hpx::serialization::output_archive& ar, unsigned)
    {
        ar << *static_cast<expr_node_type*>(this) << id;
    }

    void primary_expr::serialize(
        hpx::serialization::input_archive& ar, unsigned)
    {
        ar >> *static_cast<expr_node_type*>(this) >> id;
    }

    ///////////////////////////////////////////////////////////////////////////
    void operand::serialize(hpx::serialization::output_archive& ar, unsigned)
    {
        ar << *static_cast<operand_node_type*>(this);
    }

    void operand::serialize(hpx::serialization::input_archive& ar, unsigned)
    {
        ar >> *static_cast<operand_node_type*>(this);
    }

    ///////////////////////////////////////////////////////////////////////////
    void unary_expr::serialize(hpx::serialization::output_archive& ar, unsigned)
    {
        ar << operator_ << operand_ << id;
    }

    void unary_expr::serialize(hpx::serialization::input_archive& ar, unsigned)
    {
        ar >> operator_ >> operand_ >> id;
    }

    ///////////////////////////////////////////////////////////////////////////
    void operation::serialize(hpx::serialization::output_archive& ar, unsigned)
    {
        ar << operator_ << operand_;
    }

    void operation::serialize(hpx::serialization::input_archive& ar, unsigned)
    {
        ar >> operator_ >> operand_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void expression::serialize(hpx::serialization::output_archive& ar, unsigned)
    {
        ar << first << rest;
    }

    void expression::serialize(hpx::serialization::input_archive& ar, unsigned)
    {
        ar >> first >> rest;
    }

    ///////////////////////////////////////////////////////////////////////////
    void function_call::serialize(
        hpx::serialization::output_archive& ar, unsigned)
    {
        ar << function_name << args;
    }

    void function_call::serialize(
        hpx::serialization::input_archive& ar, unsigned)
    {
        ar >> function_name >> args;
    }

//     ///////////////////////////////////////////////////////////////////////////
//     void assignment::serialize(hpx::serialization::output_archive& ar, unsigned)
//     {
//         ar << lhs << operator_ << rhs;
//     }
//
//     void assignment::serialize(hpx::serialization::input_archive& ar, unsigned)
//     {
//         ar >> lhs >> operator_ >> rhs;
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     void variable_declaration::serialize(
//         hpx::serialization::output_archive& ar, unsigned)
//     {
//         ar << lhs << rhs;
//     }
//
//     void variable_declaration::serialize(
//         hpx::serialization::input_archive& ar, unsigned)
//     {
//         ar >> lhs >> rhs;
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     void statement::serialize(hpx::serialization::output_archive& ar, unsigned)
//     {
//         ar << *static_cast<statement_node_type*>(this);
//     }
//
//     void statement::serialize(hpx::serialization::input_archive& ar, unsigned)
//     {
//         ar >> *static_cast<statement_node_type*>(this);
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     void if_statement::serialize(
//         hpx::serialization::output_archive& ar, unsigned)
//     {
//         ar << condition << then << else_;
//     }
//
//     void if_statement::serialize(
//         hpx::serialization::input_archive& ar, unsigned)
//     {
//         ar >> condition >> then >> else_;
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     void while_statement::serialize(
//         hpx::serialization::output_archive& ar, unsigned)
//     {
//         ar << condition << body;
//     }
//
//     void while_statement::serialize(
//         hpx::serialization::input_archive& ar, unsigned)
//     {
//         ar >> condition >> body;
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     void return_statement::serialize(
//         hpx::serialization::output_archive& ar, unsigned)
//     {
//         ar << expr;
//     }
//
//     void return_statement::serialize(
//         hpx::serialization::input_archive& ar, unsigned)
//     {
//         ar >> expr;
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     void function::serialize(hpx::serialization::output_archive& ar, unsigned)
//     {
//         ar << return_type << function_name << args << body;
//     }
//
//     void function::serialize(hpx::serialization::input_archive& ar, unsigned)
//     {
//         ar >> return_type >> function_name >> args >> body;
//     }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        static constexpr char const* const optoken_names[] =
        {
            // precedence 1
            "op_comma",

            // precedence 2
            "op_assign",
            "op_plus_assign",
            "op_minus_assign",
            "op_times_assign",
            "op_divide_assign",
            "op_mod_assign",
            "op_bit_and_assign",
            "op_bit_xor_assign",
            "op_bitor_assign",
            "op_shift_left_assign",
            "op_shift_right_assign",

            // precedence 3
            "op_logical_or",

            // precedence 4
            "op_logical_and",

            // precedence 5
            "op_bit_or",

            // precedence 6
            "op_bit_xor",

            // precedence 7
            "op_bit_and",

            // precedence 8
            "op_equal",
            "op_not_equal",

            // precedence 9
            "op_less",
            "op_less_equal",
            "op_greater",
            "op_greater_equal",

            // precedence 10
            "op_shift_left",
            "op_shift_right",

            // precedence 11
            "op_plus",
            "op_minus",

            // precedence 12
            "op_times",
            "op_divide",
            "op_mod",

            // precedence 13
            "op_positive",
            "op_negative",
            "op_pre_incr",
            "op_pre_decr",
            "op_compl",
            "op_not",

            // precedence 14
            "op_post_incr",
            "op_post_decr",

            "op_unknown",
        };

        static constexpr char const* const optoken_repr[] =
        {
            // precedence 1
            ",",

            // precedence 2
            "=",
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "&=",
            "^=",
            "|=",
            "<<=",
            ">>=",

            // precedence 3
            "||",

            // precedence 4
            "&&",

            // precedence 5
            "|",

            // precedence 6
            "^",

            // precedence 7
            "&",

            // precedence 8
            "==",
            "!=",

            // precedence 9
            "<",
            "<=",
            ">",
            ">=",

            // precedence 10
            "<<",
            ">>",

            // precedence 11
            "+",
            "-",

            // precedence 12
            "*",
            "/",
            "%",

            // precedence 13
            "+",
            "-",
            "++",
            "--",
            "~",
            "!",

            // precedence 14
            "++",
            "--",

            "\?\?\?",
        };
    }

    std::ostream& operator<<(std::ostream& out, nil)
    {
        return out;
    }

    std::ostream& operator<<(std::ostream& out, optoken op)
    {
        out << detail::optoken_repr[static_cast<int>(op)];
        return out;
    }

    std::ostream& operator<<(std::ostream& out, identifier const& id)
    {
        out << id.name;
        return out;
    }

    namespace detail
    {
        struct to_string
        {
            void operator()(bool ast) const
            {
                out_ << (ast ? "true" : "false");
            }

            void operator()(std::string const& ast) const
            {
                out_ << "\"";
                for (char c : ast)
                {
                    switch (c)
                    {
                    case '\b':  out_ << "\\\b"; break;
                    case '\n':  out_ << "\\\n"; break;
                    case '\r':  out_ << "\\\r"; break;
                    case '\t':  out_ << "\\\t"; break;
                    case '\\':  out_ << "\\\\"; break;
                    case '\"':  out_ << "\\\""; break;
                    default:    out_ << c;
                    }
                }
                out_ << "\"";
            }

            template <typename Ast>
            void operator()(util::recursive_wrapper<Ast> const& ast) const
            {
                out_ << ast.get();
            }

            template <typename Ast>
            void operator()(Ast const& ast) const
            {
                out_ << ast;
            }

            std::ostream& out_;
        };
    }

    std::ostream& operator<<(std::ostream& out, primary_expr const& pe)
    {
        util::visit(detail::to_string{out}, pe.get());
        return out;
    }

    std::ostream& operator<<(std::ostream& out, operand const& op)
    {
        util::visit(detail::to_string{out}, op.get());
        return out;
    }

    std::ostream& operator<<(std::ostream& out, unary_expr const& ue)
    {
        out << ue.operator_ << ue.operand_;
        return out;
    }

    std::ostream& operator<<(std::ostream& out, operation const& op)
    {
        out << " " << op.operator_ << " " << op.operand_;
        return out;
    }

    std::ostream& operator<<(std::ostream& out, expression const& expr)
    {
        if (!expr.rest.empty())
        {
            out << "(";
        }

        out << expr.first;

        if (!expr.rest.empty())
        {
            for (auto const& op : expr.rest)
                out << op;
            out << ")";
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, function_call const& f)
    {
        out << f.function_name << "(";
        bool first = true;
        for (auto const& arg : f.args)
        {
            if (!first)
            {
                out << ", ";
            }
            first = false;
            out << arg;
        }
        out << ")";
        return out;
    }

    std::ostream& operator<<(
        std::ostream& out, std::vector<ast::expression> const& l)
    {
        out << "'(";
        bool first = true;
        for (auto const& arg : l)
        {
            if (!first)
            {
                out << ", ";
            }
            first = false;
            out << arg;
        }
        out << ")";
        return out;
    }

    std::string to_string(expression const& expr)
    {
        std::ostringstream str;
        str << expr;
        return str.str();
    }

//     std::ostream& operator<<(std::ostream& out, assignment const& a)
//     {
//         out << "assignment";
//         return out;
//     }
//
//     std::ostream& operator<<(std::ostream& out, variable_declaration const& vd)
//     {
//         out << "variable_declaration";
//         return out;
//     }
//
//     std::ostream& operator<<(std::ostream& out, statement const& stmt)
//     {
//         out << "statement";
//         return out;
//     }
//
//     std::ostream& operator<<(std::ostream& out, if_statement const& if_)
//     {
//         out << "if_statement";
//         return out;
//     }
//
//     std::ostream& operator<<(std::ostream& out, while_statement const& while_)
//     {
//         out << "while_statement";
//         return out;
//     }
//
//     std::ostream& operator<<(std::ostream& out, return_statement const& ret)
//     {
//         out << "return_statement";
//         return out;
//     }
//
//     std::ostream& operator<<(std::ostream& out, function const& func)
//     {
//         out << "function";
//         return out;
//     }
//
//     std::ostream& operator<<(std::ostream& out, function_list const& fl)
//     {
//         out << "function_list";
//         return out;
//     }
}}
