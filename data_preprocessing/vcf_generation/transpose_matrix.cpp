#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

int main() {
    // Input and output directories
    std::string data_dir = "/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/";
    std::string feature_file = data_dir + "data/features/X_pc10.txt";
    std::string output_file_path = data_dir + "data/features/subsetting/gen_different_from_0_sum_Xpc10.txt";

    // Load feature table
    std::ifstream input_file(feature_file);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open the file " << feature_file << std::endl;
        return 1;
    }
    
    // Read header line to get SNP names
    std::string header;
    std::getline(input_file, header);
    std::vector<std::string> snp_names;
    size_t start = header.find('\t') + 1; // Skip the first column name
    size_t end;
    while ((end = header.find('\t', start)) != std::string::npos) {
        snp_names.push_back(header.substr(start, end - start));
        start = end + 1;
    }
    
    // Get rid of sex and CAG columns
    snp_names.erase(snp_names.begin()); // Assuming the first column is "FID_IID"
    snp_names.erase(std::remove(snp_names.begin(), snp_names.end(), "Sex"), snp_names.end());
    snp_names.erase(std::remove(snp_names.begin(), snp_names.end(), "CAG"), snp_names.end());
    
    // Total number of samples
    size_t n_samples = 0;
    std::string line;
    while (std::getline(input_file, line))
        n_samples++;
    input_file.clear();
    input_file.seekg(0, std::ios::beg);
    
    // Initialize vector to store prevalence of alternative variant, always taking first two
    std::vector<int> row_sums(snp_names.size(), 0);
    
    // Sum all rows of each column
    while (std::getline(input_file, line)) {
        size_t pos = line.find('\t') + 1; // Skip the first column value
        for (int i = 0; i < row_sums.size(); ++i) {
            size_t next_pos = line.find('\t', pos);
            std::string val_str = line.substr(pos, next_pos - pos);
            try {
                int val = std::stoi(val_str);
                if (val != 0)
                    row_sums[i]++;
            } catch (const std::invalid_argument& e) {
                // Handle invalid argument
                std::cerr << "Invalid argument: " << e.what() << std::endl;
            }
            pos = next_pos + 1;
        }
    }
    input_file.close();
    
    // Save row_sums as a single tab-separated line in a text file
    std::ofstream output_file(output_file_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not create the file " << output_file_path << std::endl;
        return 1;
    }
    for (int i = 0; i < row_sums.size(); ++i) {
        output_file << row_sums[i];
        if (i != row_sums.size() - 1)
            output_file << '\t';
    }
    output_file.close();

    return 0;
}
