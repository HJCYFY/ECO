#include "config.hpp"

namespace config {

// TODO allow different types for a value. using template
ConfigParser::ConfigParser(const char* fname) {

    ifstream file(fname);
    if(!file.is_open())
    {
        printf("Cannot find config file!\n");
        exit(-1);
    }
	const static size_t BUFSIZE = 40960000;		// TODO overflow
	string s; s.resize(BUFSIZE);
    while (file >> s) 
    {
		if (s[0] == '#') {
			file.getline(&s[0], BUFSIZE, '\n');
			continue;
		}
        if(s == "NORMALIZE_POWER")
        {
            file >> normalize_power;
        }
        else if(s == "NORMALIZE_SIZE")
        {
            file >> normalize_size;
        }
        else if(s == "NORMALIZE_DIM")
        {
            file >> normalize_dim;
        }
        else if(s == "SEARCH_AREA_SHAPE")
        {
            file >> search_area_shape;
        }
        else if(s == "SEARCH_AREA_SCALE")
        {
            file >> search_area_scale;
        }
        else if(s == "MIN_IMG_SAMPLE_SIZE")
        {
            file >> min_image_sample_size;
        }
        else if(s == "MAX_IMG_SAMPLE_SIZE")
        {
            file >> max_image_sample_size;
        }
        else if(s == "REFINEMENT_ITE")
        {
            file >> refinement_iterations;
        }
        else if(s == "NEWTON_ITE")
        {
            file >> newton_iterations;
        }
        else if(s == "CLAMP_POS")
        {
            file >> clamp_position;
        }
        else if(s == "OUTPUT_SIGMA_FACTOR")
        {
            file >> output_sigma_factor;
        }
        else if(s == "LEARNING_RATE")
        {
            file >> learning_rate;
        }
        else if(s == "NSAMPLES")
        {
            file >> nSamples;
        }
        // else if(s == "SAMPLE_REPLACE_STRATEGY")
        // {
        //     file >> sample_replace_strategy;
        // }
        else if(s == "LT_SIZE")
        {
            file >> lt_size;
        }
        else if(s == "TRAIN_GAP")
        {
            file >> train_gap;
        }
        else if(s == "SKIP_AFTER_FRAME")
        {
            file >> skip_after_frame;
        }
        // else if(s == "USE_DETECTION_SAMPLE")
        // {
        //     file >> use_detection_sample;
        // }
        else if(s == "USE_PROJECTION_MATRIX")
        {
            file >> use_projection_matrix;
        }
        else if(s == "UPDATE_PROJECTION_MATRIX")
        {
            file >> update_projection_matrix;
        }
        else if(s == "PROJ_INIT_METHOD")
        {
            file >> proj_init_method;
        }
        else if(s == "PROJECTION_REG")
        {
            file >> projection_reg;
        }
        else if(s == "USE_SAMPLE_MERGE")
        {
            file >> use_sample_merge;
        }
        else if(s == "SAMPLE_MERGE_TYPE")
        {
            file >> sample_merge_type;
        }
        // else if(s == "DISTANCE_MATRIX_UPDATA_TYPE")
        // {
        //     file >> distance_matrix_update_type;
        // }
        else if(s == "CG_ITER")
        {
            file >> CG_iter;
        }
        else if(s == "INIT_CG_ITER")
        {
            file >> init_CG_iter;
        }
        else if(s == "INIT_GN_ITER")
        {
            file >> init_GN_iter;
        }
        else if(s == "CG_USE_FR")
        {
            file >> CG_use_FR;
        }
        else if(s == "CG_STANDARD_ALPHA")
        {
            file >> CG_standard_alpha;
        }
        else if(s == "CG_FORGETTING_RATE")
        {
            file >> CG_forgetting_rate;
        }
        else if(s == "PRECOND_DATA_PARAM")
        {
            file >> precond_data_param;
        }
        else if(s == "PRECOND_REG_PARAM")
        {
            file >> precond_reg_param;
        }
        else if(s == "PRACOND_PROJ_PARAM")
        {
            file >> precond_proj_param;
        }
        else if(s == "USE_REG_WINDOW")
        {
            file >> use_reg_window;
        }
        else if(s == "REG_WINDOW_MIN")
        {
            file >> reg_window_min;
        }
        else if(s == "REG_WINDOW_EDGE")
        {
            file >> reg_window_edge;
        }
        else if(s == "REG_WINDOW_POWER")
        {
            file >> reg_window_power;
        }
        else if(s == "REG_SPARSITY_THRESGOLD")
        {
            file >> reg_sparsity_threshold;
        }
        else if(s == "INTERPOLATION_METHOD")
        {
            file >> interpolation_method;
        }
        else if(s == "INTERPOLATION_BICUBIC_A")
        {
            file >> interpolation_bicubic_a;
        }
        else if(s == "INTERPOLATION_CENTERING")
        {
            file >> interpolation_centering;
        }
        else if(s == "INTERPOLATION_WINDOWING")
        {
            file >> interpolation_windowing;
        }
        else if(s == "NUMBER_OF_SCALES")
        {
            file >> number_of_scales;
        }
        else if(s == "SCALE_STEP")
        {
            file >> scale_step;
        }
        else if(s == "USE_SCALE_FILTER")
        {
            file >> use_scale_filter;
        }
        else if(s == "SCALE_SIGMA_FACTOR")
        {
            file >> scale_sigma_factor;
        }
        else if(s == "SCALE_LEARNING_RATE")
        {
            file >> scale_learning_rate;
        }
        else if(s == "NUMBER_OF_SCALES_FILTER")
        {
            file >> number_of_scales_filter;
        }
        else if(s == "NUMBER_OF_INTERP_SCALES")
        {
            file >> number_of_interp_scales;
        }
        else if(s == "SCALE_MODEL_FACTOR")
        {
            file >> scale_model_factor;
        }
        else if(s == "SCALE_STEP_FILTER")
        {
            file >> scale_step_filter;
        }
        else if(s == "SCALE_MODEL_MAX_AREA")
        {
            file >> scale_model_max_area;
        }
        // else if(s == "SCALE_FEATURE")
        // {
        //     file >> scale_feature;
        // }
        else if(s == "S_NUM_COMPRESSED_DIM")
        {
            file >> s_num_compressed_dim;
        }
        else if(s == "LAMBDA")
        {
            file >> lambda;
        }
        else if(s == "DO_POLY_INTERP")
        {
            file >> do_poly_interp;
        }
        else if(s == "VISUALIZATION")
        {
            file >> visualization;
        }
        else if(s == "DEBUG")
        {
            file >> debug;
        }
        else if(s == "HOG_CELL_SIZE")
        {
            file >> hog_cell_size;
        }
        else if(s == "CN_CELL_SIZE")
        {
            file >> cn_cell_size;
        }
        else if(s == "HOG_ORIENT_NUM")
        {
            file >> hog_orient_num;
        }
        else if(s == "HOG_COMPRESSED_DIM")
        {
            file >> hog_compressed_dim;
        }
        else if(s == "CN_COMPRESSED_DIM")
        {
            file >> cn_compressed_dim;
        }
        else if(s == "SIZE_MODE")
        {
            file >> size_mode;
        }
        else if(s == "DEBUG1")
        {
            file >> debug1;
        }
        else if(s == "DEBUG2")
        {
            file >> debug2;
        }
        else if(s == "DEBUG3")
        {
            file >> debug3;
        }
        else if(s == "DEBUG4")
        {
            file >> debug4;
        }
        else if(s == "DEBUG5")
        {
            file >> debug5;
        }
        else if(s == "DEBUG6")
        {
            file >> debug6;
        }
        else if(s == "DEBUG7")
        {
            file >> debug7;
        }
        else if(s == "DEBUG8")
        {
            file >> debug8;
        }
        else if(s == "FILE_NAME")
        {
            file >> file_name;
        }
        else
        {
            // cout<<s<<endl;
        }
		file.getline(&s[0], BUFSIZE, '\n');
	}
}

// Global feature parameters1s
int normalize_power = 2;
bool normalize_size = true;
bool normalize_dim = true;

// Image sample parameters
int search_area_shape = 1;    // The shape of the samples
float search_area_scale = 4.0;         // The scaling of the target size to get the search area
float min_image_sample_size = 150*150;   // Minimum area of image samples
float max_image_sample_size = 200*200;   // Maximum area of image samples

// Detection parameters
int refinement_iterations = 1;       // Number of iterations used to refine the resulting position in a frame
int newton_iterations = 5;           // The number of Newton iterations used for optimizing the detection score
bool clamp_position = false;          // Clamp the target position to be inside the image

// Learning parameters
float output_sigma_factor = 1.0f/16.0f;		// Label function sigma
float learning_rate = 0.009;	 	 	// Learning rate
int nSamples = 30;                   // Maximum number of stored training samples
// string sample_replace_strategy = "lowest_prior";    // Which sample to replace when the memory is full
float lt_size = 0;                     // The size of the long-term memory (where all samples have equal weight)
int train_gap = 5;                   // The number of intermediate frames with no training (0 corresponds to training every frame)
int skip_after_frame = 10;           // After which frame number the sparse update scheme should start (1 is directly)
// bool use_detection_sample = true;     // Use the sample that was extracted at the detection stage also for learning

// Factorized convolution parameters
bool use_projection_matrix = true;    // Use projection matrix, i.e. use the factorized convolution formulation
bool update_projection_matrix = true; // Whether the projection matrix should be optimized or not
int proj_init_method = 0;        // Method for initializing the projection matrix
float projection_reg = 1e-7;	 	 	// Regularization paremeter of the projection matrix

// Generative sample space model parameters
bool use_sample_merge = true;                 // Use the generative sample space model to merge samples
int sample_merge_type = 1;             // Strategy for updating the samples (0 for 'replace' ,1 for 'merge')
// int distance_matrix_update_type = 1;   // Strategy for updating the distance matrix

// Conjugate Gradient parameters
int CG_iter = 5;                     // The number of Conjugate Gradient iterations in each update after the first frame
int init_CG_iter = 10*15;            // The total number of Conjugate Gradient iterations used in the first frame
int init_GN_iter = 10;               // The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
bool CG_use_FR = false;               // Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
bool CG_standard_alpha = true;        // Use the standard formula for computing the step length in Conjugate Gradient
float CG_forgetting_rate = 50;	 	 	// Forgetting rate of the last conjugate direction
float precond_data_param = 0.75;       // Weight of the data term in the preconditioner
float precond_reg_param = 0.25;	 	// Weight of the regularization term in the preconditioner
float precond_proj_param = 40;	 	 	// Weight of the projection matrix part in the preconditioner

// Regularization window parameters
bool use_reg_window = true;           // Use spatial regularization or not
float reg_window_min = 1e-4;			// The minimum value of the regularization window
float reg_window_edge  = 10e-3;         // The impact of the spatial regularization
float reg_window_power = 2;            // The degree of the polynomial to use (e.g. 2 is a quadratic window)
float reg_sparsity_threshold = 0.05;   // A relative threshold of which DFT coefficients that should be set to zero

// Interpolation parameters
int interpolation_method = 2;    // The kind of interpolation kernel
float interpolation_bicubic_a = -0.75;     // The parameter for the bicubic interpolation kernel
bool interpolation_centering = true;      // Center the kernel at the feature sample
bool interpolation_windowing = false;     // Do additional windowing on the Fourier coefficients of the kernel

// Scale parameters for the translation model
// Only used if: float use_scale_filter = false
int number_of_scales = 7;            // Number of scales to run the detector
float scale_step = 1.01;               // The scale factor

// Scale filter parameters
// Only used if: float use_scale_filter = true

// set use_scale_filter to true
bool use_scale_filter = true;         // Use the fDSST scale filter or not (for speed)
float scale_sigma_factor = 1.0f/16.0f;       // Scale label function sigma
float scale_learning_rate = 0.025;		// Scale filter learning rate
int number_of_scales_filter = 17;    // Number of scales
int number_of_interp_scales = 33;    // Number of interpolated scales
float scale_model_factor = 1.0;        // Scaling of the scale model
float scale_step_filter = 1.02;        // The scale factor for the scale filter
float scale_model_max_area = 32*16;    // Maximume area for the scale sample patch
// string scale_feature = "HOG4";          // Features for the scale filter (only HOG4 supported)
int s_num_compressed_dim = -1;    // Number of compressed feature dimensions in the scale filter {-1 max}
float lambda = 1e-2;					// Scale filter regularization
bool do_poly_interp = true;           // Do 2nd order polynomial interpolation to obtain more accurate scale

// Visualization
bool visualization = true;               // Visualiza tracking and detection scores
bool debug = false;                       // Do full debug visualization

// feature extract
int hog_cell_size = 6; 
int cn_cell_size = 4; 
int hog_orient_num = 9; 

int hog_compressed_dim = 10;
int cn_compressed_dim = 3;
int size_mode = 2;

int debug1 = 0;
int debug2 = 0;
int debug3 = 0;
int debug4 = 0;
int debug5 = 0;
int debug6 = 0;
int debug7 = 0;
int debug8 = 0;

string file_name = "Bird2";
}