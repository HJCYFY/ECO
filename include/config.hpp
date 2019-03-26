#pragma once
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
using namespace std;

namespace config {
class ConfigParser {
public:
    ConfigParser(const char* fname);
};


// Global feature parameters1s
extern int normalize_power;
extern bool normalize_size;
extern bool normalize_dim;

// Image sample parameters
extern int search_area_shape ;    // The shape of the samples
extern float search_area_scale ;         // The scaling of the target size to get the search area
extern float min_image_sample_size ;   // Minimum area of image samples
extern float max_image_sample_size ;   // Maximum area of image samples

// Detection parameters
extern int refinement_iterations ;       // Number of iterations used to refine the resulting position in a frame
extern int newton_iterations ;           // The number of Newton iterations used for optimizing the detection score
extern bool clamp_position ;          // Clamp the target position to be inside the image

// Learning parameters
extern float output_sigma_factor;		// Label function sigma
extern float learning_rate;	 	 	// Learning rate
extern int nSamples;                   // Maximum number of stored training samples
// extern string sample_replace_strategy;    // Which sample to replace when the memory is full
extern float lt_size ;                     // The size of the long-term memory (where all samples have equal weight)
extern int train_gap ;                   // The number of intermediate frames with no training (0 corresponds to training every frame)
extern int skip_after_frame ;           // After which frame number the sparse update scheme should start (1 is directly)
// extern bool use_detection_sample ;     // Use the sample that was extracted at the detection stage also for learning

// Factorized convolution parameters
extern bool use_projection_matrix;    // Use projection matrix, i.e. use the factorized convolution formulation
extern bool update_projection_matrix; // Whether the projection matrix should be optimized or not
extern int proj_init_method;        // Method for initializing the projection matrix
extern float projection_reg;	 	 	// Regularization paremeter of the projection matrix

// Generative sample space model parameters
extern bool use_sample_merge;                 // Use the generative sample space model to merge samples
extern int sample_merge_type;             // Strategy for updating the samples

// Conjugate Gradient parameters
extern int CG_iter;                     // The number of Conjugate Gradient iterations in each update after the first frame
extern int init_CG_iter;            // The total number of Conjugate Gradient iterations used in the first frame
extern int init_GN_iter;               // The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
extern bool CG_use_FR;               // Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
extern bool CG_standard_alpha;        // Use the standard formula for computing the step length in Conjugate Gradient
extern float CG_forgetting_rate;	 	 	// Forgetting rate of the last conjugate direction
extern float precond_data_param;       // Weight of the data term in the preconditioner
extern float precond_reg_param;	 	// Weight of the regularization term in the preconditioner
extern float precond_proj_param;	 	 	// Weight of the projection matrix part in the preconditioner

// Regularization window parameters
extern bool use_reg_window ;           // Use spatial regularization or not
extern float reg_window_min ;			// The minimum value of the regularization window
extern float reg_window_edge ;         // The impact of the spatial regularization
extern float reg_window_power ;            // The degree of the polynomial to use (e.g. 2 is a quadratic window)
extern float reg_sparsity_threshold ;   // A relative threshold of which DFT coefficients that should be set to zero

// Interpolation parameters
extern int interpolation_method ;    // The kind of interpolation kernel
extern float interpolation_bicubic_a ;     // The parameter for the bicubic interpolation kernel
extern bool interpolation_centering ;      // Center the kernel at the feature sample
extern bool interpolation_windowing ;     // Do additional windowing on the Fourier coefficients of the kernel

// Scale parameters for the translation model
// Only used if: float use_scale_filter = false
extern int number_of_scales ;            // Number of scales to run the detector
extern float scale_step ;               // The scale factor

// Scale filter parameters
// Only used if: float use_scale_filter = true
extern bool use_scale_filter ;         // Use the fDSST scale filter or not (for speed)
extern float scale_sigma_factor ;       // Scale label function sigma
extern float scale_learning_rate ;		// Scale filter learning rate
extern int number_of_scales_filter ;    // Number of scales
extern int number_of_interp_scales ;    // Number of interpolated scales
extern float scale_model_factor ;        // Scaling of the scale model
extern float scale_step_filter ;        // The scale factor for the scale filter
extern float scale_model_max_area ;    // Maximume area for the scale sample patch
// extern string scale_feature ;          // Features for the scale filter (only HOG4 supported)
extern int s_num_compressed_dim ;    // Number of compressed feature dimensions in the scale filter
extern float lambda ;					// Scale filter regularization
extern bool do_poly_interp ;           // Do 2nd order polynomial interpolation to obtain more accurate scale

// Visualization
extern bool visualization;               // Visualiza tracking and detection scores
extern bool debug;                       // Do full debug visualization

extern int hog_cell_size; 
extern int cn_cell_size; 
extern int hog_orient_num; 

extern int hog_compressed_dim;
extern int cn_compressed_dim;

extern int size_mode;

extern int debug1;
extern int debug2;
extern int debug3;
extern int debug4;
extern int debug5;
extern int debug6;
extern int debug7;
extern int debug8;

extern string file_name;
}