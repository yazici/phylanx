// Copyright (c) 2017-2018 Monil, Mohammad Alaul Haque
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)



#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/mean_operation.hpp>
#include <phylanx/ir/node_data.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>
#include <hpx/throw_exception.hpp>

#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <blaze/Math.h>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace execution_tree { namespace primitives
        {
            ///////////////////////////////////////////////////////////////////////////
            primitive create_mean_operation(hpx::id_type const& locality,
                                            std::vector<primitive_argument_type>&& operands,
                                                   std::string const& name, std::string const& codename)
            {
                static std::string type("mean");
                return create_primitive_component(
                        locality, type, std::move(operands), name, codename);
            }

            match_pattern_type const mean_operation::match_data =
            {
                    hpx::util::make_tuple("mean",
                                          std::vector<std::string>{"mean(_1, _2)"},
                                          &create_mean_operation,
                                          &create_primitive<mean_operation>)
            };

            ///////////////////////////////////////////////////////////////////////////
            mean_operation::mean_operation(
                    std::vector<primitive_argument_type>&& operands,
                    std::string const& name, std::string const& codename)
                    : primitive_component_base(std::move(operands), name, codename)
            {}

            ///////////////////////////////////////////////////////////////////////////


            primitive_argument_type mean_operation::mean0d(args_type&& args) const
            {
                // `axis` is optional
                if (args.size() == 2)
                {
                    // `axis` must be a scalar if provided
                    if (args[1].num_dimensions() != 0)
                    {
                        HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                            "mean_operation::mean0d",
                                            execution_tree::generate_error_message(
                                                    "operand axis must be a scalar", name_,
                                                    codename_));
                    }
                    const int axis = args[1].scalar();
                    // `axis` can only be -1 or 0
                    if (axis < -1 || axis > 0)
                    {
                        HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                            "mean_operation::mean0d",
                                            execution_tree::generate_error_message(
                                                    "operand axis can only between -1 and 0 for "
                                                            "an a operand that is 0d",
                                                    name_, codename_));
                    }
                }
                return primitive_argument_type{std::move(args[0])};
            }

            ///////////////////////////////////////////////////////////////////////////
            primitive_argument_type mean_operation::mean1d(args_type&& args) const
            {
                // `axis` is optional
                if (args.size() == 2)
                {
                    // `axis` must be a scalar if provided
                    if (args[1].num_dimensions() != 0)
                    {
                        HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                            "mean_operation::mean1d",
                                            execution_tree::generate_error_message(
                                                    "operand axis must be a scalar", name_,
                                                    codename_));
                    }
                    const int axis = args[1].scalar();
                    // `axis` can only be -1 or 0
                    if (axis < -1 || axis > 0)
                    {
                        HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                            "mean_operation::mean1d",
                                            execution_tree::generate_error_message(
                                                    "operand axis can only between -1 and 0 for "
                                                            "an a operand that is 1d",
                                                    name_, codename_));
                    }
                }

                auto a = args[0].vector();

                // a should not be empty
                if (a.size() == 0)
                {
                    HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                        "mean_operation::mean1d",
                                        execution_tree::generate_error_message(
                                                "attempt to get argmin of an empty sequence",
                                                name_, codename_));
                }

                // Find the sum of all the elements
                const auto sum = std::accumulate(a.begin(), a.end(), 0.0);

                // Return the mean
                return sum / a.size();
            }


            hpx::future<primitive_argument_type> mean_operation::eval(
                    std::vector<primitive_argument_type> const& operands,
                    std::vector<primitive_argument_type> const& args) const
            {
                if (operands.size() != 1 && operands.size() != 2)
                {
                    HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                        "phylanx::execution_tree::primitives::"
                                                "mean_operation::mean_operation",
                                        execution_tree::generate_error_message(
                                                "the mean_operation primitive requires "
                                                        "either one or two arguments",
                                                name_, codename_));
                }

                bool arguments_valid = true;
                for (std::size_t i = 0; i != operands.size(); ++i)
                {
                    if (!valid(operands[i]))
                    {
                        arguments_valid = false;
                    }
                }

                if (!arguments_valid)
                {
                    HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                        "mean_operation::eval",
                                        execution_tree::generate_error_message(
                                                "the mean_operation primitive requires "
                                                        "that the arguments given by the operands "
                                                        "array are valid",
                                                name_, codename_));
                }

                auto this_ = this->shared_from_this();
                return hpx::dataflow(hpx::util::unwrapping(
                        [this_](args_type&& args) -> primitive_argument_type
                        {
                            std::size_t matrix_dims = args[0].num_dimensions();
                            switch (matrix_dims)
                            {
                                case 0:
                                    return this_->mean0d(std::move(args));

                                case 1:
                                    return this_->mean1d(std::move(args));

                                case 2:
                                    return this_->mean0d(std::move(args));

                                default:
                                    HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                                        "mean_operation::eval",
                                                        execution_tree::generate_error_message(
                                                                "left hand side operand has unsupported "
                                                                        "number of dimensions",
                                                                this_->name_, this_->codename_));
                            }
                        }), detail::map_operands(
                                             operands, functional::numeric_operand{}, args,
                                             name_, codename_));
            }

            //////////////////////////////////////////////////////////////////////////
            hpx::future<primitive_argument_type> mean_operation::eval(
                    std::vector<primitive_argument_type> const& args) const
            {
                if (operands_.empty())
                {
                    return eval(args, noargs);
                }
                return eval(operands_, args);
            }
        }}}