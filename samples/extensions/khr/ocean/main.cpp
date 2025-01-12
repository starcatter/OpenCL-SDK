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

#include <GL/glew.h>

#include <iostream>
#include <valarray>
#include <random>
#include <algorithm>
#include <fstream>
#include <tuple>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tclap/CmdLine.h>

#include "ocean.hpp"

void OceanApplication::event(const sf::Event& event)
{
    switch (event.type)
    {
        case sf::Event::Closed: close(); break;
        case sf::Event::Resized:
            glViewport(0, 0, getSize().x, getSize().y);
            checkError("glViewport(0, 0, getSize().x, getSize().y)");
            break;
        case sf::Event::KeyPressed:
            keyboard(event.key.code);
            break;
        case sf::Event::MouseButtonPressed:
            if (event.mouseButton.button == sf::Mouse::Button::Left)
            {
                camera.drag = true;
                camera.begin = glm::vec2(event.mouseButton.x, event.mouseButton.y);
            }
            break;
        case sf::Event::MouseButtonReleased:
            if (event.mouseButton.button == sf::Mouse::Button::Left)
                camera.drag = false;
            break;
        case sf::Event::MouseMoved:
            if (camera.drag)
                mouseDrag(event.mouseMove.x, event.mouseMove.y);
            break;
        case sf::Event::MouseWheelMoved:
            camera.eye += camera.dir * (float)event.mouseWheel.delta * ROLL_SPEED_FAC;
            break;
        default: break;
    }
samples.back().end = clock_type::now();
}

void OceanApplication::mouseDrag(const int x, const int y)
{
    if (!camera.drag) return;

    glm::vec2 off = camera.begin - glm::vec2(x, y);
    camera.begin = glm::vec2(x, y);

    camera.yaw -= off.x * DRAG_SPEED_FAC;
    camera.pitch += off.y * DRAG_SPEED_FAC;

    glm::quat yaw(glm::cos(glm::radians(camera.yaw / 2)),
                  glm::vec3(0, 0, 1) * glm::sin(glm::radians(camera.yaw / 2)));
    glm::quat pitch(glm::cos(glm::radians(camera.pitch / 2)),
                    glm::vec3(1, 0, 0)
                        * glm::sin(glm::radians(camera.pitch / 2)));
    glm::mat3 rot_mat(yaw * pitch);
    glm::vec3 dir = rot_mat * glm::vec3(0, 0, -1);

    camera.dir = glm::normalize(dir);
    camera.rvec = glm::normalize(glm::cross(camera.dir, glm::vec3(0, 0, 1)));
    camera.up = glm::normalize(glm::cross(camera.rvec, camera.dir));
}

void OceanApplication::keyboard(int key)
{
    switch (key)
    {
        case sf::Keyboard::Key::Escape:
            close();
            break;
        case sf::Keyboard::Key::Space:
            animate = !animate;
            printf("animation is %s\n", animate ? "ON" : "OFF");
            break;

        case sf::Keyboard::Key::A:
            wind_magnitude += 1.f;
            changed = true;
            break;
        case sf::Keyboard::Key::Z:
            wind_magnitude -= 1.f;
            changed = true;
            break;

        case sf::Keyboard::Key::S:
            wind_angle += 1.f;
            changed = true;
            break;
        case sf::Keyboard::Key::X:
            wind_angle -= 1.f;
            changed = true;
            break;

        case sf::Keyboard::Key::D:
            amplitude += 0.5f;
            changed = true;
            break;
        case sf::Keyboard::Key::C:
            amplitude -= 0.5f;
            changed = true;
            break;

        case sf::Keyboard::Key::F: choppiness += 0.5f; break;
        case sf::Keyboard::Key::V: choppiness -= 0.5f; break;

        case sf::Keyboard::Key::G: alt_scale += 0.5f; break;
        case sf::Keyboard::Key::B: alt_scale -= 0.5f; break;

        case sf::Keyboard::Key::W:
            wireframe_mode = !wireframe_mode;
            break;

        case sf::Keyboard::Key::E:
            show_fps = !show_fps;
            break;
    }
}

void OceanApplication::show_fps_window_title()
{
    if (show_fps)
    {
        auto fps_now = std::chrono::system_clock::now();

        std::chrono::duration<float> elapsed = fps_now - fps_last_time;
        float delta = elapsed.count();

        const float elapsed_tres = 1.f;

        delta_frames++;
        if (delta >= 1.f)
        {
            double fps = double(delta_frames) / delta;

            std::stringstream ss;
            ss << app_name << ", [FPS:" << std::fixed << std::setprecision(2)
               << fps << "]";

            setTitle(ss.str().c_str());

            delta_frames = 0;
            fps_last_time = fps_now;
        }
    }
    else
    {
        fps_last_time = std::chrono::system_clock::now();
        delta_frames = 0;
    }
}


cl_platform_id getPlatformFromDevice(cl_device_id deviceID)
{
    cl_platform_id platform = nullptr;
    cl_int err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platform),
                                 &platform, nullptr);
    return platform;
}

void OceanApplication::initializeCL()
{
    auto device = opencl_context.getInfo<CL_CONTEXT_DEVICES>().at(0);

    command_queue = cl::CommandQueue{ opencl_context, device/*, CL_QUEUE_PROFILING_ENABLE*/ };

    if (use_cl_khr_gl_sharing
        && cl::util::supports_extension(device, "cl_khr_gl_sharing"))
    {
        std::cout << "cl_khr_gl_sharing supported" << std::endl;
    }
    else
    {
        std::cout << "cl_khr_gl_sharing not supported" << std::endl;
        use_cl_khr_gl_sharing = false;
    }

    int error = CL_SUCCESS;
    error |= clGetDeviceInfo(
        device(), CL_DEVICE_IMAGE2D_MAX_WIDTH,
        sizeof(ocl_max_img2d_width), &ocl_max_img2d_width, NULL);
    error |= clGetDeviceInfo(
        device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(ocl_max_alloc_size), &ocl_max_alloc_size, NULL);
    error |= clGetDeviceInfo(device(),
                             CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ocl_mem_size),
                             &ocl_mem_size, NULL);

    if (error != CL_SUCCESS) printf("clGetDeviceInfo error: %d\n", error);

    auto build_opencl_kernel = [&](const char* src_file, cl::Kernel& kernel,
                                   const char* name) {
        try
        {
            std::string kernel_code =
                cl::util::read_exe_relative_text_file(src_file);
            cl::Program program{ opencl_context, kernel_code };
            program.build();
            kernel = cl::Kernel{ program, name };
        } catch (const cl::BuildError& e)
        {
            auto bl = e.getBuildLog();
            std::cout << "Build OpenCL " << name
                      << " kernel error: " << std::endl;
            for (auto& elem : bl) std::cout << elem.second << std::endl;
            exit(1);
        }
    };

    build_opencl_kernel("twiddle.cl", twiddle_kernel, "generate");
    build_opencl_kernel("init_spectrum.cl", init_spectrum_kernel,
                        "init_spectrum");
    build_opencl_kernel("time_spectrum.cl", time_spectrum_kernel, "spectrum");
    build_opencl_kernel("fft_kernel.cl", fft_kernel, "fft_1D");
    build_opencl_kernel("inversion.cl", inversion_kernel, "inversion");
    build_opencl_kernel("normals.cl", normals_kernel, "normals");

    // init opencl resources
    try
    {
        {
            std::vector<cl_float4> phase_array(ocean_tex_size * ocean_tex_size);
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_real_distribution<float> dist(0.f, 1.f);

            for (size_t i = 0; i < phase_array.size(); ++i)
                phase_array[i] = { dist(rng), dist(rng), dist(rng), dist(rng) };

            noise_mem = std::make_unique<cl::Image2D>(
                opencl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), ocean_tex_size,
                ocean_tex_size, 0, phase_array.data());
        }

        hkt_pong_mem = std::make_unique<cl::Image2D>(
            opencl_context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
            ocean_tex_size, ocean_tex_size);

        dxyz_coef_mem[0] = std::make_unique<cl::Image2D>(
            opencl_context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
            ocean_tex_size, ocean_tex_size);

        dxyz_coef_mem[1] = std::make_unique<cl::Image2D>(
            opencl_context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
            ocean_tex_size, ocean_tex_size);

        dxyz_coef_mem[2] = std::make_unique<cl::Image2D>(
            opencl_context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
            ocean_tex_size, ocean_tex_size);

        h0k_mem = std::make_unique<cl::Image2D>(
            opencl_context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT),
            ocean_tex_size, ocean_tex_size);

        size_t log_2_N = (size_t)((log((float)ocean_tex_size) / log(2.f)) - 1);

        twiddle_factors_mem = std::make_unique<cl::Image2D>(
            opencl_context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT),
            log_2_N, ocean_tex_size);

        for (size_t target = 0; target < IOPT_COUNT; target++)
        {
            if (use_cl_khr_gl_sharing)
            {
                ocl_image_mems[target].reset(new cl::Image2D{
                    clCreateFromGLTexture(opencl_context(), CL_MEM_READ_WRITE,
                                          GL_TEXTURE_2D, 0,
                                          texture_images[target], NULL) });
            }
            else
            {
                ocl_image_mems[target].reset(
                    new cl::Image2D{ opencl_context, CL_MEM_READ_WRITE,
                                     cl::ImageFormat{ CL_RGBA, CL_FLOAT },
                                     ocean_tex_size, ocean_tex_size });
            }
        }
    } catch (const cl::Error& e)
    {
        printf("initOpenCLMems: OpenCL %s image error: %s\n", e.what(),
               IGetErrorString(e.err()));
        exit(1);
    }
}

void OceanApplication::update_spectrum(float elapsed)
{
    cl_int2 patch =
        cl_int2{ (int)(ocean_grid_size * mesh_spacing), (int)ocean_tex_size };

    cl::NDRange lws; // NullRange by default.
    if (group_size > 0)
    {
        lws = cl::NDRange{ group_size, group_size };
    }

    if (twiddle_factors_init)
    {
        try
        {
            size_t log_2_N =
                (size_t)((log((float)ocean_tex_size) / log(2.f)) - 1);

            /// Prepare vector of values to extract results
            std::vector<cl_int> v(ocean_tex_size);
            for (int i = 0; i < ocean_tex_size; i++)
            {
                int x = reverse_bits(i, log_2_N);
                v[i] = x;
            }

            /// Initialize device-side storage
            cl::Buffer bit_reversed_inds_mem{
                opencl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(cl_int) * v.size(), v.data()
            };

            twiddle_kernel.setArg(0, cl_int(ocean_tex_size));
            twiddle_kernel.setArg(1, bit_reversed_inds_mem);
            twiddle_kernel.setArg(2, *twiddle_factors_mem);

            command_queue.enqueueNDRangeKernel(
                twiddle_kernel, cl::NullRange,
                cl::NDRange{ (cl::size_type)log_2_N, ocean_tex_size },
                cl::NDRange{ 1, 16 });
            twiddle_factors_init = false;
        } catch (const cl::Error& e)
        {
            printf("twiddle indices: OpenCL %s kernel error: %s\n", e.what(),
                   IGetErrorString(e.err()));
            exit(1);
        }
    }

    // change of some ocean's parameters requires to rebuild initial spectrum
    // image
    if (changed)
    {
        try
        {
            float wind_angle_rad = glm::radians(wind_angle);
            cl_float4 params =
                cl_float4{ wind_magnitude * glm::cos(wind_angle_rad),
                           wind_magnitude * glm::sin(wind_angle_rad), amplitude,
                           supress_factor };
            init_spectrum_kernel.setArg(0, patch);
            init_spectrum_kernel.setArg(1, params);
            init_spectrum_kernel.setArg(2, *noise_mem);
            init_spectrum_kernel.setArg(3, *h0k_mem);

            command_queue.enqueueNDRangeKernel(
                init_spectrum_kernel, cl::NullRange,
                cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);
            changed = false;
        } catch (const cl::Error& e)
        {
            printf("initial spectrum: OpenCL %s kernel error: %s\n", e.what(),
                   IGetErrorString(e.err()));
            exit(1);
        }
    }

    // ping-pong phase spectrum kernel launch
    try
    {
        time_spectrum_kernel.setArg(0, elapsed);
        time_spectrum_kernel.setArg(1, patch);
        time_spectrum_kernel.setArg(2, *h0k_mem);
        time_spectrum_kernel.setArg(3, *dxyz_coef_mem[0]);
        time_spectrum_kernel.setArg(4, *dxyz_coef_mem[1]);
        time_spectrum_kernel.setArg(5, *dxyz_coef_mem[2]);

        command_queue.enqueueNDRangeKernel(
            time_spectrum_kernel, cl::NullRange,
            cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws, nullptr);
    } catch (const cl::Error& e)
    {
        printf("updateSpectrum: OpenCL %s kernel error: %s\n", e.what(),
               IGetErrorString(e.err()));
        exit(1);
    }


    // perform 1D FFT horizontal and vertical iterations
    size_t log_2_N = (size_t)((log((float)ocean_tex_size) / log(2.f)) - 1);
    fft_kernel.setArg(1, patch);
    fft_kernel.setArg(2, *twiddle_factors_mem);
    for (cl_int i = 0; i < 3; i++)
    {
        const cl::Image* displ_swap[] = { dxyz_coef_mem[i].get(),
                                          hkt_pong_mem.get() };
        cl_int2 mode = cl_int2{ { 0, 0 } };

        bool ifft_pingpong = false;
        for (int p = 0; p < log_2_N; p++)
        {
            if (ifft_pingpong)
            {
                fft_kernel.setArg(3, *displ_swap[1]);
                fft_kernel.setArg(4, *displ_swap[0]);
            }
            else
            {
                fft_kernel.setArg(3, *displ_swap[0]);
                fft_kernel.setArg(4, *displ_swap[1]);
            }

            mode.s[1] = p;
            fft_kernel.setArg(0, mode);

            command_queue.enqueueNDRangeKernel(
                fft_kernel, cl::NullRange,
                cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);

            ifft_pingpong = !ifft_pingpong;
        }

        // Cols
        mode.s[0] = 1;
        for (int p = 0; p < log_2_N; p++)
        {
            if (ifft_pingpong)
            {
                fft_kernel.setArg(3, *displ_swap[1]);
                fft_kernel.setArg(4, *displ_swap[0]);
            }
            else
            {
                fft_kernel.setArg(3, *displ_swap[0]);
                fft_kernel.setArg(4, *displ_swap[1]);
            }

            mode.s[1] = p;
            fft_kernel.setArg(0, mode);

            command_queue.enqueueNDRangeKernel(
                fft_kernel, cl::NullRange,
                cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);

            ifft_pingpong = !ifft_pingpong;
        }

        if (log_2_N % 2)
        {
            // swap images if pingpong hold on temporary buffer
            std::array<size_t, 3> orig = { 0, 0, 0 },
                                  region = { ocean_tex_size, ocean_tex_size,
                                             1 };
            command_queue.enqueueCopyImage(*displ_swap[0], *displ_swap[1], orig,
                                           orig, region);
        }
    }

    if(use_cl_khr_gl_sharing)
    {
        for (size_t target = 0; target < texture_images.size(); target++)
            clEnqueueAcquireGLObjects(command_queue(), 1,
                                      &(*ocl_image_mems[target])(), 0, nullptr,
                                      nullptr);
    }

    // inversion
    {
        inversion_kernel.setArg(0, patch);
        inversion_kernel.setArg(1, *dxyz_coef_mem[0]);
        inversion_kernel.setArg(2, *dxyz_coef_mem[1]);
        inversion_kernel.setArg(3, *dxyz_coef_mem[2]);
        inversion_kernel.setArg(4, *ocl_image_mems[IOPT_DISPLACEMENT]);

        command_queue.enqueueNDRangeKernel(
            inversion_kernel, cl::NullRange,
            cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws);
    }

    // normals computation
    {
        cl_float2 factors = cl_float2{ choppiness, alt_scale };

        normals_kernel.setArg(0, patch);
        normals_kernel.setArg(1, factors);
        normals_kernel.setArg(2, *noise_mem);
        normals_kernel.setArg(3, *ocl_image_mems[IOPT_DISPLACEMENT]);
        normals_kernel.setArg(4, *ocl_image_mems[IOPT_NORMAL_MAP]);

        command_queue.enqueueNDRangeKernel(
            normals_kernel, cl::NullRange,
            cl::NDRange{ ocean_tex_size, ocean_tex_size }, lws, nullptr);
    }

    if(use_cl_khr_gl_sharing)
    {
        if (cl_khr_gl_event_supported == false)
            command_queue.finish();
        for (size_t target = 0; target < texture_images.size(); target++)
            clEnqueueReleaseGLObjects(command_queue(), 1,
                                      &(*ocl_image_mems[target])(), 0,
                                      nullptr, nullptr);
    }
    else
    {
        for (size_t target = 0; target < texture_images.size(); target++)
        {
            size_t rowPitch = 0;
            void* pixels = command_queue.enqueueMapImage(
                *ocl_image_mems[target], CL_TRUE, CL_MAP_READ, { 0, 0, 0 },
                { ocean_tex_size, ocean_tex_size, 1 }, &rowPitch, nullptr);

            glBindTexture(GL_TEXTURE_2D, texture_images[target]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)ocean_tex_size,
                         (GLsizei)ocean_tex_size, 0, GL_RGBA, GL_FLOAT, pixels);

            command_queue.enqueueUnmapMemObject(*ocl_image_mems[target],
                                                pixels);
        }
    }
}

template <> auto cl::sdk::parse<CliOptions>()
{
    return std::make_tuple(
        std::make_shared<TCLAP::ValueArg<bool>>("", "useGLSharing",
                                                "Use cl_khr_gl_sharing",
                                                false, true, "boolean"));
}

template <>
CliOptions cl::sdk::comprehend<CliOptions>(
    std::shared_ptr<TCLAP::ValueArg<bool>> useGLSharing)
{
    return CliOptions{
        useGLSharing->getValue()
    };
}

int main(int argc, char* argv[])
{
    auto start = clock_type::now();

    OceanApplication app;

    try
    {
        // Parse command-line options
        auto opts =
            cl::sdk::parse_cli<cl::sdk::options::Diagnostic,
                               cl::sdk::options::SingleDevice, CliOptions>(argc, argv);
        const auto& dev_opts = std::get<1>(opts);

        app.dev_opts = dev_opts;
        app.use_cl_khr_gl_sharing = std::get<2>(opts).use_gl_sharing;
        app.num_instances = std::get<2>(opts).num_instaces;

        app.run();
    } catch (cl::util::Error& e)
    {
        std::cerr << "OpenCL Utils error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (cl::BuildError& e)
    {
        std::cerr << "OpenCL runtime error: " << e.what() << std::endl;
        for (auto& build_log : e.getBuildLog())
        {
            std::cerr << "\tBuild log for device: "
                      << build_log.first.getInfo<CL_DEVICE_NAME>() << "\n"
                      << std::endl;
            std::cerr << build_log.second << "\n" << std::endl;
        }
        std::exit(e.err());
    } catch (cl::Error& e)
    {
        std::cerr << "OpenCL runtime error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }





    auto end = clock_type::now();
    // generate file name
    auto in_time_t = std::chrono::system_clock::to_time_t(end);

    std::stringstream ss;
    ss << "timings_gl_interop_";
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X");
    ss << ".csv";

    const auto filename = ss.str();

    app.save_results(filename);




    return 0;
}
