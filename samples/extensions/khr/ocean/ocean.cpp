/*
 * Copyright (c) 2024 Mobica Limited, Marcin Hajder, Piotr Plebański
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <random>
#include <set>
#include <stdexcept>

#include <math.h>

// GLM includes
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ocean.hpp"

void OceanApplication::initializeGL()
{
    if (glewInit() != GLEW_OK) std::exit(EXIT_FAILURE);

    window_width = getSize().x;
    window_height = getSize().y;

    auto create_shader = [](std::string file_path, cl_GLenum shader_stage) {
        std::string shader_string =
            cl::util::read_exe_relative_text_file(file_path.c_str());
        auto pshader_string = shader_string.c_str();
        GLuint shader = glCreateShader(shader_stage);
        glShaderSource(shader, 1, &pshader_string, NULL);
        glCompileShader(shader);

        GLint status = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE)
        {
            int log_length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
            std::vector<GLchar> shader_log(log_length);
            glGetShaderInfoLog(shader, log_length, NULL, shader_log.data());
            std::cerr << std::string(shader_log.cbegin(), shader_log.cend())
                      << std::endl;
        }

        return shader;
    };
    auto create_program = [](std::initializer_list<GLuint> shader_stages) {
        GLuint program = glCreateProgram();
        for (auto shader_stage : shader_stages)
        {
            glAttachShader(program, shader_stage);
        }

        glLinkProgram(program);
        GLint status = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &status);
        if (status != GL_TRUE)
        {
            int log_length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
            std::vector<GLchar> program_log(log_length);
            glGetProgramInfoLog(program, log_length, NULL, program_log.data());
            std::cerr << std::string(program_log.cbegin(), program_log.cend())
                      << std::endl;
        }

        return program;
    };

    cl_GLuint vertex_shader = create_shader("ocean.vert", GL_VERTEX_SHADER);
    cl_GLuint fragment_shader = create_shader("ocean.frag", GL_FRAGMENT_SHADER);
    gl_program = create_program({ vertex_shader, fragment_shader });

    glUseProgram(gl_program);

    create_texture_images();
    create_vertex_buffer();
    create_index_buffer();
    create_uniform_buffer();

    glViewport(0, 0, getSize().x, getSize().y);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE);
}

using test_clock = std::chrono::high_resolution_clock;
std::chrono::system_clock::time_point start_perf;
std::chrono::system_clock::time_point end_perf;
const int max_perf_count=100;
void OceanApplication::updateScene()
{
    show_fps_window_title();

static float perf_avg=0.f;
static int perf_cnt=0;
start_perf = test_clock::now();


    update_uniforms();

    auto end = std::chrono::system_clock::now();

    // time factor of ocean animation
    static float elapsed = 0.f;

    if (animate)
    {
        std::chrono::duration<float> delta = end - start;
        elapsed = delta.count();
        update_spectrum(elapsed);
    }
    else
    {
        // hold the animation at the same time point
        std::chrono::duration<float> duration(elapsed);
        start =
            end - std::chrono::duration_cast<std::chrono::seconds>(duration);
        command_queue.finish();
    }

end_perf = test_clock::now();
std::chrono::duration<float> elapsed_seconds = end_perf - start_perf;
perf_avg+=elapsed_seconds.count();
perf_cnt++;
if(perf_cnt==max_perf_count)
{

    printf("chrono time OpenCL processing: %f\n", perf_avg/perf_cnt);
    perf_cnt=0;
    perf_avg=0.f;
}
}

void OceanApplication::render()
{
static float perf_avg=0.f;
static int perf_cnt=0;
start_perf = test_clock::now();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(gl_program);
    glBindVertexArray(vertex_array);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

    for (size_t target = 0; target < IOPT_COUNT; target++)
    {
        glActiveTexture(GL_TEXTURE0 + target);
        glBindTexture(GL_TEXTURE_2D, texture_images[target]);
    }

    if (wireframe_mode)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glDrawElementsInstanced(GL_TRIANGLE_STRIP, ocean_grid_indices.size(),
                            GL_UNSIGNED_INT, 0, num_instances);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Wait for all drawing commands to finish
    if (!cl_khr_gl_event_supported)
        glFinish();
    else
        glFlush();

end_perf = test_clock::now();
std::chrono::duration<float> elapsed_seconds = end_perf - start_perf;
perf_avg+=elapsed_seconds.count();
perf_cnt++;
if(perf_cnt==max_perf_count)
{

    printf("chrono time OpenGL processing: %f\n", perf_avg/perf_cnt);
    perf_cnt=0;
    perf_avg=0.f;
}
}

void OceanApplication::cleanup()
{
    if ( vertex_buffer!=0 )
    {
        glDeleteBuffers(1, &vertex_buffer);
        vertex_buffer=0;
    }

    if ( index_buffer!=0 )
    {
        glDeleteBuffers(1, &index_buffer);
        index_buffer=0;
    }

    if(vertex_array!=0)
    {
        glDeleteVertexArrays(1, &vertex_array);
        vertex_array = 0;
    }

    if(gl_program!=0)
    {
        glDeleteProgram(gl_program);
        gl_program=0;
    }

    for (size_t target = 0; target < texture_images.size(); target++)
    {
        glDeleteTextures(1, &texture_images[target]);
    }
}

void OceanApplication::create_uniform_buffer()
{
    glGenBuffers(1, &view_data_ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, view_data_ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformBufferObject), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void OceanApplication::create_vertex_buffer()
{
    size_t iCXY = (ocean_grid_size + 1) * (ocean_grid_size + 1);
    ocean_grid_vertices.resize(iCXY);

    cl_float dfY = -0.5f * (ocean_grid_size * mesh_spacing),
             dfBaseX = -0.5f * (ocean_grid_size * mesh_spacing);
    cl_float tx = 0.f, ty = 0.f, dtx = 1.f / ocean_grid_size,
             dty = 1.f / ocean_grid_size;
    for (size_t iBase = 0, iY = 0; iY <= ocean_grid_size;
         iY++, iBase += ocean_grid_size + 1)
    {
        tx = 0.f;
        cl_float dfX = dfBaseX;
        for (int iX = 0; iX <= ocean_grid_size; iX++)
        {
            *((cl_float4*)&ocean_grid_vertices[iBase + iX].pos) = {dfX, dfY, 0.f, 0.f};
            *((cl_float2*)&ocean_grid_vertices[iBase + iX].tc) = {tx, ty};
            tx += dtx;
            dfX += mesh_spacing;
        }
        dfY += mesh_spacing;
        ty += dty;
    }

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, ocean_grid_vertices.size() * sizeof(Vertex),
                 ocean_grid_vertices.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    const GLsizei vertSize = sizeof(Vertex);
    const GLsizei uvOffset = sizeof(Vertex::pos);

    // Setup shader attributes
    GLint attribPos = glGetAttribLocation(gl_program, "in_position");
    if(attribPos != -1) {
        glVertexAttribPointer(attribPos, 4, GL_FLOAT, GL_FALSE, vertSize, nullptr);
        glEnableVertexAttribArray(attribPos);
    }
    else std::cerr << "Shader attribute not valid" << std::endl;

    GLint attribUV = glGetAttribLocation(gl_program, "in_tex_coords");
    if(attribUV != -1) {
        glVertexAttribPointer(attribUV, 2, GL_FLOAT, GL_FALSE, vertSize, (const GLvoid *)uvOffset);
        glEnableVertexAttribArray(attribUV);
    }
    else std::cerr << "Shader attribute not valid" << std::endl;

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OceanApplication::create_index_buffer()
{
    size_t totalIndices = ((ocean_grid_size + 1) * 2 + 1) * ocean_grid_size;
    ocean_grid_indices.resize(totalIndices);

    size_t indexCount = 0;
    for (size_t iY = 0; iY < ocean_grid_size; iY++)
    {
        size_t iBaseFrom = iY * (ocean_grid_size + 1);
        size_t iBaseTo = iBaseFrom + ocean_grid_size + 1;

        for (size_t iX = 0; iX <= ocean_grid_size; iX++)
        {
            ocean_grid_indices[indexCount++] = static_cast<int>(iBaseFrom + iX);
            ocean_grid_indices[indexCount++] = static_cast<int>(iBaseTo + iX);
        }
        ocean_grid_indices[indexCount++] = -1;
    }

    glEnable(GL_PRIMITIVE_RESTART);

    glPrimitiveRestartIndex(-1);
    glGenBuffers(1, &index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(ocean_grid_indices.front()) * ocean_grid_indices.size(),
                 ocean_grid_indices.data(), GL_STATIC_DRAW);
}

void OceanApplication::create_texture_images()
{
    std::string uni_names[] = {"u_displacement_map", "u_normal_map"};
    for (size_t target = 0; target < texture_images.size(); target++)
    {
        texture_images[target] = 0;
        glGenTextures(1, &texture_images[target]);
        glBindTexture(GL_TEXTURE_2D, texture_images[target]);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)ocean_tex_size,
                     (GLsizei)ocean_tex_size, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        GLint uniformLocation = glGetUniformLocation(gl_program, uni_names[target].c_str());

        if (uniformLocation!=-1)
        {
            glUniform1i(uniformLocation, target);
        }
    }
}

void OceanApplication::update_uniforms()
{
    UniformBufferObject ubo = {};
    ubo.choppiness = choppiness;
    ubo.alt_scale = alt_scale;

    // update camera related uniform
    glm::mat4 view_matrix =
        glm::lookAt(camera.eye, camera.eye + camera.dir, camera.up);

    float fov = (float)glm::radians(60.0);
    float aspect = (float)window_width / window_height;
    glm::mat4 proj_matrix = glm::perspective(
        fov, aspect, 1.f, 2.f * ocean_grid_size * mesh_spacing);

    ubo.view_mat = view_matrix;
    ubo.proj_mat = proj_matrix;

    glBindBuffer(GL_UNIFORM_BUFFER, view_data_ubo);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UniformBufferObject), &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    GLuint blockIndex = glGetUniformBlockIndex(gl_program, "ViewData");
    glUniformBlockBinding(gl_program, blockIndex, 2);
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, view_data_ubo);
}
