//
// Created by feng on 19-4-16.
//

#include "NewTypeHeliostatForTest.h"

int NewTypeHeliostatForTest::getSubHelioSize() {
    return row_col_.x * row_col_.y;
}

void NewTypeHeliostatForTest::setSize(float3 size) {
    size_ = size;
}

std::vector<float> NewTypeHeliostatForTest::getSurfaceProperty() {
    return std::vector<float>(6, -1);
}

void NewTypeHeliostatForTest::CGetSubHeliostatVertexes(std::vector<float3> &SubHeliostatVertexes) {
    printf("test here.....\n");
    float2 subhelio_row_col_length;
    subhelio_row_col_length.x = (size_.z - gap_.y * (row_col_.x - 1)) / float(row_col_.x);
    subhelio_row_col_length.y = (size_.x - gap_.x * (row_col_.y - 1)) / float(row_col_.y);
    std::cout << "subhelio_row_col_length--------" << subhelio_row_col_length.x << ", " << subhelio_row_col_length.y << std::endl;
    std::cout << "subhelio_row_col_length--------" << subhelio_row_col_length.x << ", " << subhelio_row_col_length.y << std::endl;
    float3 row_direction = normalize(vertex_[0] - vertex_[1]);
    float3 column_direction = normalize(vertex_[2] - vertex_[1]);

    std::cout << "row_direction--------" << row_direction.x << ", " << row_direction.y << ", " << row_direction.z << std::endl;
    std::cout << "col_direction--------" << column_direction.x << ", " << column_direction.y << ", " << column_direction.z << std::endl;

    for (int r = 0; r < row_col_.x; ++r) {
        for (int c = 0; c < row_col_.y; ++c) {
            float relativeColumnLength = c * (gap_.x + subhelio_row_col_length.y);
            float relativeRowLength = r * (gap_.y + subhelio_row_col_length.x);
            float3 v1 = vertex_[1] + row_direction * relativeRowLength + column_direction * relativeColumnLength;
            float3 v0 = v1 + row_direction * subhelio_row_col_length.x;
            float3 v2 = v1 + column_direction * subhelio_row_col_length.y;
            std::cout << "v0--------" << v0.x << ", " << v0.y << ", " << v0.z << std::endl;
            std::cout << "v1--------" << v1.x << ", " << v1.y << ", " << v1.z << std::endl;
            std::cout << "v2--------" << v2.x << ", " << v2.y << ", " << v2.z << std::endl;
            SubHeliostatVertexes.push_back(v0);
            SubHeliostatVertexes.push_back(v1);
            SubHeliostatVertexes.push_back(v2);
        }
    }
}