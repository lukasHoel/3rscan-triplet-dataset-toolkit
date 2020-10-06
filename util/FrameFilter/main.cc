#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <fstream>
#include "json11.hpp"

#include <Eigen/Dense>

#include <set>

struct Data {
    std::string scan_id;
    std::string reference_id;
    int instance_id;
    std::string frame_id;
    std::string semantic_label;
    Eigen::Vector4i bounding_box;
    std::vector<float> visibility;
    std::vector<int> other_instance_ids;
    std::vector<Eigen::Vector4i> other_bboxes;

    Data(const std::string& scan_id, const std::string& reference_id, const std::string frame_id,
         const int instance, const std::string& semantic_label,
         const Eigen::Vector4i& bb, std::vector<float>& visibility, 
         const std::vector<int>& other_instance_ids, std::vector<Eigen::Vector4i>& other_bboxes): scan_id(scan_id), reference_id(reference_id), frame_id(frame_id),
                                     instance_id(instance), semantic_label(semantic_label),
                                     bounding_box(bb), visibility(visibility),
                                     other_instance_ids(other_instance_ids), other_bboxes(other_bboxes) {
    }
};

void load_scan3R(const std::string& file, std::map<std::string, std::string>& scan2reference) {
    std::cout << "load " << file << std::endl;
    std::string err;
    std::ifstream is(file);
    std::string dataset((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    json11::Json json = json11::Json::parse(dataset, err);
    for (auto &scene: json.array_items()) {
        const std::string reference_id = scene["reference"].string_value();
        scan2reference[reference_id] = reference_id;
        for (auto &scan: scene["scans"].array_items()) {
            const std::string scan_id = scan["reference"].string_value();
            scan2reference[scan_id] = reference_id;
        }
    }
}

void load_semseg(const std::string& semseg, std::map<int, std::string>& instance2label) {
    std::string err;
    std::ifstream is(semseg);
    std::string dataset((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    json11::Json json = json11::Json::parse(dataset, err);
    for (auto &k: json["segGroups"].array_items()) {
        const int id = k["id"].number_value();
        const std::string label = k["label"].string_value();
        instance2label[id] = label;
    }
}

void load_label_filter(std::set<std::string>& label_filter) {
    std::ifstream file("../filter_labels.txt");
    std::string line;
    if (file.is_open()){
        while(getline(file, line)){
            label_filter.insert(line);
        }
    }
}

bool is_valid_class(const std::string& semantic_label, std::set<std::string>& label_filter) {
    return label_filter.find(semantic_label) != label_filter.end();
}

bool is_bigbb(const Eigen::Vector4i& box) {
    return (box[2] - box[0]) * (box[3] - box[1]) > (256*256);
}

void load_visibility(std::map<int, std::vector<float>>& visibility, const std::string& rendered_path, const std::string& frame_id){
    std::ifstream file(rendered_path + "frame-" + frame_id + ".visibility.txt");
    std::string line;
    if (file.is_open()){
        while(getline(file, line)){
            std::stringstream ss(line);
            int instance_id;
            int truncation_number_pixels_original_image, truncation_number_pixels_larger_fov_image;
            float truncation;
            int occlusion_number_pixels_original_image, occlusion_number_pixels_only_with_that_instance;
            float occlusion;

            ss >> instance_id;
            ss >> truncation_number_pixels_original_image >> truncation_number_pixels_larger_fov_image >> truncation;
            ss >> occlusion_number_pixels_original_image >> occlusion_number_pixels_only_with_that_instance >> occlusion;

            visibility[instance_id] = {truncation_number_pixels_original_image, truncation_number_pixels_larger_fov_image, truncation,
                                       occlusion_number_pixels_original_image, occlusion_number_pixels_only_with_that_instance, occlusion};
        }
    }
}

bool is_visible_enough(std::vector<float>& visibility) {
    int truncation_number_pixels_original_image = visibility[0];
    int truncation_number_pixels_larger_fov_image = visibility[1];
    float truncation = visibility[2];

    int occlusion_number_pixels_original_image = visibility[3];
    int occlusion_number_pixels_only_with_that_instance = visibility[4];
    float occlusion = visibility[5];

    return truncation_number_pixels_original_image > 0 && truncation_number_pixels_larger_fov_image > 0 && truncation > 0.6
        && occlusion_number_pixels_original_image > 0 && occlusion_number_pixels_only_with_that_instance > 0 && occlusion > 0.6;
}

bool is_big_mask(std::vector<float>& visibility) {
    int occlusion_number_pixels_original_image = visibility[3];
    return occlusion_number_pixels_original_image > 256*256;
}

bool has_visibility(std::map<int, std::vector<float>> visibility, int instance_id){
    // It can happen that we have an empty map (no visibility.txt file present)
    // It can happen that we have an instance in .bb.txt but not in .visibility.txt (no entry in map at position instance_id)
    return !visibility.empty() && visibility.count(instance_id) && !visibility[instance_id].empty();
}

void evaluate_frame(const std::string& rendered_path,
             const std::string& bbox_file_path,
             const std::string& scan_id,
             const std::string& reference_id,
             const std::string& frame_id,
             const std::map<int, std::string>& instance2label,
             std::vector<Data>& data,
             std::set<std::string>& labels,
             std::set<std::string>& label_filter) {
    std::string line;
    std::ifstream bbox_file(bbox_file_path);

    // stores all instance ids and bboxes in this picture
    std::vector<int> instance_ids;
    std::vector<Eigen::Vector4i> bboxes;

    //loop file to collect all instance ids and bboxes
    if (bbox_file.is_open()) {
        while (getline(bbox_file, line)) {
            std::stringstream ss(line);
            std::vector<std::string> tokens;
            std::string token;
            while(std::getline(ss, token, ' ')){
                tokens.emplace_back(token);
            }
            int l = tokens.size();

            int x1 = std::stoi(tokens[l - 4]);
            int y1 = std::stoi(tokens[l - 3]);
            int x2 = std::stoi(tokens[l - 2]);
            int y2 = std::stoi(tokens[l - 1]);
            Eigen::Vector4i box(x1, y1, x2, y2);
            bboxes.emplace_back(box);

            int instance_id = std::stoi(tokens[0]);
            instance_ids.emplace_back(instance_id);
        }
    }

    // sanity check if file was loaded correctly
    assert(instance_ids.size() == bboxes.size());

    // load visibility
    std::map<int, std::vector<float>> visibility;
    load_visibility(visibility, rendered_path, frame_id);

    //loop data to find interesting objects
    for(int i=0; i<instance_ids.size(); i++){
        Eigen::Vector4i box = bboxes[i];
        int instance_id = instance_ids[i];

        // if the instance is valid (should always be the case)
        if (instance2label.find(instance_id) != instance2label.end()) {
            const std::string semantic_label = instance2label.at(instance_id);
            labels.insert(semantic_label);

            // if this instance fulfills all filter-criteria to be a valid anchor
            if (is_valid_class(semantic_label, label_filter) && is_bigbb(box) && has_visibility(visibility, instance_id) && is_visible_enough(visibility[instance_id]) && is_big_mask(visibility[instance_id])){
                // find other instance ids / bboxes visible in this frame for the current selected instance
                std::vector<int> other_instance_ids;
                std::vector<Eigen::Vector4i> other_bboxes;
                for(int k=0; k<instance_ids.size(); k++){
                    Eigen::Vector4i other_bbox = bboxes[k];
                    int other_instance_id = instance_ids[k];

                    // only add instace ids of interesting class because we are not interested to know that there is e.g. also a wall in this image
                    if(other_instance_id != instance_id && instance2label.find(other_instance_id) != instance2label.end() && is_valid_class(instance2label.at(other_instance_id), label_filter)){
                        other_instance_ids.emplace_back(other_instance_id);
                        other_bboxes.emplace_back(other_bbox);
                    }
                }

                // fill data
                data.push_back(Data(scan_id, reference_id, frame_id, instance_id, semantic_label, box, visibility[instance_id], other_instance_ids, other_bboxes));
            }
        }
    }
}

bool compareData(const Data &a, const Data &b) {
    if (a.reference_id == b.reference_id) {
        if (a.scan_id == b.scan_id) {
            return a.instance_id < b.instance_id;
        } else
            return a.scan_id < b.scan_id;
    } else {
        return a.reference_id < b.reference_id;
    }
}

void save_data(std::vector<Data>& data, const std::string& file_path) {
    std::sort(data.begin(), data.end(), compareData);
    std::ofstream file(file_path);
    if (file.is_open()) {
        file << "reference\tscan_id\tframe_id\tinstance_id\tsemantic_label\tbounding box\tvisibility\tother_instance_ids_and_bboxes" << std::endl;
        for (const auto& data_elem: data) {
            file << data_elem.reference_id << "\t" <<
                    data_elem.scan_id << "\t" <<
                    data_elem.frame_id << "\t" <<
                    data_elem.instance_id << "\t" <<
                    data_elem.semantic_label << "\t" <<
                    data_elem.bounding_box(0) << "\t" <<
                    data_elem.bounding_box(1) << "\t" <<
                    data_elem.bounding_box(2) << "\t" <<
                    data_elem.bounding_box(3) << "\t" <<
                    data_elem.visibility[0] << "\t" <<
                    data_elem.visibility[1] << "\t" <<
                    data_elem.visibility[2] << "\t" <<
                    data_elem.visibility[3] << "\t" <<
                    data_elem.visibility[4] << "\t" <<
                    data_elem.visibility[5] << "\t" <<
                    data_elem.other_instance_ids.size();

            for(int i=0; i<data_elem.other_instance_ids.size(); i++){
                file << "\t" << data_elem.other_instance_ids[i];
                file << "\t" << data_elem.other_bboxes[i](0);
                file << "\t" << data_elem.other_bboxes[i](1);
                file << "\t" << data_elem.other_bboxes[i](2);
                file << "\t" << data_elem.other_bboxes[i](3);
            }
            file << std::endl;
        }
    }
    file.close();
}

void save_labels(std::set<std::string>& labels, const std::string& file_path) {
    std::ofstream file(file_path);
    if (file.is_open()) {
        for (const auto& label: labels) {
            file << label << std::endl;
        }
    }
    file.close();
}

void iterate(const std::string& data_path) {
    std::vector<Data> data;
    std::set<std::string> labels;
    std::set<std::string> label_filter;
    load_label_filter(label_filter);
    std::map<std::string, std::string> scan2reference;
    load_scan3R(data_path + "/3RScan.json", scan2reference);

    const std::filesystem::path path{data_path};
    std::map<std::string, int> database; // key needs to be <reference_id>_<instance_id> frame_id
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
       const auto scan_id = entry.path().filename().string();
       if (entry.is_directory()) {
           const std::string seq_path = data_path + "/" + scan_id + "/rendered/";
           // we need to load the semseg json file to get the labels.
           const std::string semseg = data_path + "/" + scan_id + "/semseg.v2.json";
           std::map<int, std::string> instance2label;
           load_semseg(semseg, instance2label);
           // if our sequence folder exists and our instances are not empty
           if (std::filesystem::exists(seq_path) && !instance2label.empty()) {
               const std::filesystem::path seq_path_fs{seq_path};
               std::cout << "Going through: " << seq_path << std::endl;
               for (const auto& entry_files: std::filesystem::directory_iterator(seq_path_fs)) {
                   if (entry_files.is_regular_file()) {
                       const auto filenameStr = entry_files.path().filename().string();
                       if (filenameStr.find("bb.txt") != std::string::npos) {
                           const std::string frame_id = filenameStr.substr(6,6);
                           evaluate_frame(seq_path, seq_path + "/" + filenameStr, scan_id, scan2reference[scan_id],
                                   frame_id, instance2label, data, labels, label_filter);
                       }
                   }
               }
           }
       }
    }
    save_data(data, data_path + "/2DInstances.txt");
    save_labels(labels, data_path + "/labels.txt");

    std::cout << "Selected " << data.size() << " instances" << std::endl;
}

int main(int argc, char **argv) {
    if (argc > 2)
        return -1;
    const std::string data_path = argv[1];
    iterate(data_path);
    return 0;
}
