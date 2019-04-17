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
    float2 subhelio_row_col_length;
    subhelio_row_col_length.x = (size_.z - gap_.y * (row_col_.x - 1)) / float(row_col_.x);
    subhelio_row_col_length.y = (size_.x - gap_.x * (row_col_.y - 1)) / float(row_col_.y);

    float3 row_direction = normalize(vertex_[0] - vertex_[1]);
    float3 column_direction = normalize(vertex_[2] - vertex_[1]);

    for (int r = 0; r < row_col_.x; ++r) {
        for (int c = 0; c < row_col_.y; ++c) {
            float relativeColumnLength = c * (gap_.x + subhelio_row_col_length.y);
            float relativeRowLength = r * (gap_.y + subhelio_row_col_length.x);
            float3 v1 = vertex_[1] + row_direction * relativeRowLength + column_direction * relativeColumnLength;
            float3 v0 = v1 + row_direction * subhelio_row_col_length.x;
            float3 v2 = v1 + column_direction * subhelio_row_col_length.y;

            SubHeliostatVertexes.push_back(v0);
            SubHeliostatVertexes.push_back(v1);
            SubHeliostatVertexes.push_back(v2);
        }
    }
}