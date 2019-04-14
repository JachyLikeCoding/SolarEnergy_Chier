//
// Created by feng on 19-4-10.
//

#include <iostream>
#include <string>

#include "RegularExpressionTree.h"
#include "gtest/gtest.h"

class RegularExpressionTreeFixture : public ::testing::Test{
public:
    SceneRegularExpressionTree sceneTree;

    bool check_expression(std::string expression){
        std::cout << "The expression is: '" << expression << "'" << std::endl;
        TreeNode *node = sceneTree.getRoot();
        int i = 0;
        try{
            for(; i < expression.size(); ++i){
                node = sceneTree.step_forward(node, expression[i]);
            }
            sceneTree.check_terminated(node);
            return true;
        }catch(std::runtime_error e){
            std::cerr << e.what() << "Error occurs at position " << i << " in expression: "
                    << expression <<"." << std::endl;
            return false;
        }
    }
};

TEST_F(RegularExpressionTreeFixture, goodExample){
    //D(R(GH+)+)+
    std::string goodExample1("DRGH");
    std::string goodExample2("DRGHH");
    std::string goodExample3("DRGHGH");
    std::string goodExample4("DRGHHGHH");
    std::string goodExample5("DRGHRGH");

    EXPECT_TRUE(check_expression(goodExample1));
    EXPECT_TRUE(check_expression(goodExample2));
    EXPECT_TRUE(check_expression(goodExample3));
    EXPECT_TRUE(check_expression(goodExample4));
}

TEST_F(RegularExpressionTreeFixture, badExampleOfIncorrectInput){
    std::string emptyExample("");
    std::string lostExample("DR");
    std::string duplicateExample1("DRRGHGH");
    std::string duplicateExample2("DRGGHHRGHH");

    EXPECT_FALSE(check_expression(emptyExample));
    EXPECT_FALSE(check_expression(lostExample));
    EXPECT_FALSE(check_expression(duplicateExample1));
    EXPECT_FALSE(check_expression(duplicateExample2));
}

TEST_F(RegularExpressionTreeFixture, badExampleOfInvalidInput) {
    // D(R(GH+)+)+
    std::string InvalidExample1("drgh");
    std::string InvalidExample2("DRG");
    std::string InvalidExample3("*DRGHRGH");

    EXPECT_FALSE(check_expression(InvalidExample1));
    EXPECT_FALSE(check_expression(InvalidExample2));
    EXPECT_FALSE(check_expression(InvalidExample3));
}