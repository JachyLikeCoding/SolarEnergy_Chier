//
// Created by feng on 19-4-8.
//

#ifndef SOLARENERGY_CHIER_REGULAREXPRESSIONTREE_H
#define SOLARENERGY_CHIER_REGULAREXPRESSIONTREE_H

#include <vector>
#define NEXT_SIZE 26


class TreeNode{
public:
    TreeNode():terminated_signal(false){
        init_next();
    }

    TreeNode(bool terminated):terminated_signal(terminated){
        init_next();
    }

    TreeNode *getNextNode(char c);
    bool setNextNode(char c, TreeNode *nextNode);

    bool isTerminated() const;
    void setTerminatedSignal(bool terminated_signal);


private:
    bool terminated_signal;
    std::vector<TreeNode *> next;
    void init_next();
};



class SceneRegularExpressionTree{
public:
    SceneRegularExpressionTree(){
        setUpTree();
    }

    ~SceneRegularExpressionTree(){
        destroyTree();
    }

    TreeNode *getRoot(){
        return start_node;
    }

    TreeNode *step_forward(TreeNode *node, char c);

    void check_terminated(TreeNode *node);



private:
    bool setUpTree();
    bool destroyTree();
    TreeNode *start_node;
    TreeNode *ground_node;
    TreeNode *receiver_node;
    TreeNode *grid_node;
    TreeNode *heliostat_node;
    TreeNode *subheliostat_node;

};

#endif //SOLARENERGY_CHIER_REGULAREXPRESSIONTREE_H
