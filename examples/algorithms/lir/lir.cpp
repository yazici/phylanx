//   Copyright (c) 2018 Tianyi Zhang
//
//   Distributed under the Boost Software License, Version 1.0. (See accompanying
//   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/phylanx.hpp>
#include <hpx/hpx_init.hpp>

#include <cstdint>
#include <iostream>

#include <boost/program_options.hpp>
#include <blaze/Math.h>

///////////////////////////////////////////////////////////////////////////////
char const* const lir_code = R"(block(
    //
    // Linear regression analysis algorithm
    //
    //   x: [30, 2]
    //   y: [30]
    define(lir, x, y, alpha, iterations, enable_output,
        block(
            define(weights, constant(0.0, shape(x, 1))),         // weights: [2]
            define(transx, transpose(x)),                        // transx:  [2, 30]
            define(pred, constant(0.0, shape(x, 0))),
            define(error, constant(0.0, shape(x, 0))),
            define(gradient, constant(0.0, shape(x, 1))),
            define(step, 0),
            while(
                step < iterations,
                block(
                    if(enable_output, cout("step: ", step, ", ", weights)),
                    store(pred, dot(x, weights)),
                    store(error, pred - y),                      // error: [30]
                    store(gradient, dot(transx, error)),         // gradient: [2]
                    parallel_block(
                        store(weights, weights - (alpha * gradient)),
                        store(step, step + 1)
                    )
                )
            ),
            weights
        )
    ),
    lir
))";

int hpx_main(boost::program_options::variables_map& vm)
{
    blaze::DynamicMatrix<double> v1{
            {15.04, 16.74}, {13.82, 24.49}, {12.54, 16.32}, {23.09, 19.83},
            {9.268, 12.87}, {9.676, 13.14}, {12.22, 20.04}, {11.06, 17.12},
            {16.3, 15.7}, {15.46, 23.95}, {11.74, 14.69}, {14.81, 14.7},
            {13.4, 20.52}, {14.58, 13.66}, {15.05, 19.07}, {11.34, 18.61},
            {18.31, 20.58}, {19.89, 20.26}, {12.88, 18.22}, {12.75, 16.7},
            {9.295, 13.9}, {24.63, 21.6}, {11.26, 19.83}, {13.71, 18.68},
            {9.847, 15.68}, {8.571, 13.1}, {13.46, 18.75}, {12.34, 12.27},
            {13.94, 13.17}, {12.07, 13.44}};

    blaze::DynamicVector<double> v2{1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
                                    0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1};

    // compile the given code
    phylanx::execution_tree::compiler::function_list snippets;
    auto const& code =
        phylanx::execution_tree::compile("lir", lir_code, snippets);
    auto lir = code.run();

    // evaluate generated execution tree
    auto x = phylanx::ir::node_data<double>{v1};
    auto y = phylanx::ir::node_data<double>{v2};
    auto alpha = phylanx::ir::node_data<double>{1e-4};

    auto iterations = vm["num_iterations"].as<std::int64_t>();
    bool enable_output = vm.count("enable_output") != 0;

    auto result = lir(x, y, alpha, iterations, enable_output);

    std::cout << "Result: \n"
              << phylanx::execution_tree::extract_numeric_value(result)
              << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // command line handling
    boost::program_options::options_description desc("usage: lir [options]");
    desc.add_options()
            ("enable_output,e", "enable progress output (default: false)")
            ("num_iterations,n",
             boost::program_options::value<std::int64_t>()->default_value(850),
             "number of iterations (default: 850)")
            ;

    return hpx::init(desc, argc, argv);
}

