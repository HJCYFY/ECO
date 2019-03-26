#include "sample_space.hpp"

namespace Track{

SampleSpace::SampleSpace():num_training_samples(0),
                        gram_matrix(config::nSamples,config::nSamples),
                        distance_matrix(config::nSamples,config::nSamples)
{
    vec_zero = VectorXf::Zero(config::nSamples);
    prior_weights = VectorXf::Zero(config::nSamples);
    float* gram_matrix_data = gram_matrix.data();
    for(int i=0;i<gram_matrix.size();++i)
        gram_matrix_data[i] = INFINITY;

    float* distance_matrix_data = distance_matrix.data();
    for(int i=0;i<distance_matrix.size();++i)
        distance_matrix_data[i] = INFINITY;
}

/******************************
 * 参与的变量 samplesf xlf_proj_perm distance_matrix gram_matrix prior_weights num_training_samples
 * 更改的变量 merged_sample merged_sample_id new_sample_id distance_matrix gram_matrix prior_weights
 * 后期 会将 merged_sample_id new_sample_id merged_sample 删除， 将samplesf的更新放在内部
 * ***************************/
void SampleSpace::update_sample_space_model(const Array4D<complex<float> >& cn_samplesf,const Array4D<complex<float> >& hog_samplesf,
                            const Array4D<complex<float> >& cn_train_sample,const Array4D<complex<float> >& hog_train_sample)
{
    VectorXf gram_vector = find_gram_vector(cn_samplesf,hog_samplesf,cn_train_sample,hog_train_sample);
    float new_train_sample_norm = (cn_train_sample.getVector().array().abs2().sum() + hog_train_sample.getVector().array().abs2().sum())*2;
    
    merged_sample_id = -1;
    new_sample_id = -1;

    float min_sample_weight;
    int min_sample_idr,min_sample_idc;

    if(num_training_samples >= config::nSamples)
    {   
        min_sample_weight = prior_weights.minCoeff(&min_sample_idr, &min_sample_idc);
        if(min_sample_weight < minimum_sample_weight)
        {
            update_distance_matrix(gram_vector,new_train_sample_norm,min_sample_idr,-1,0,1);
            prior_weights(min_sample_idr) = 0;
            prior_weights *= (1 - config::learning_rate)/prior_weights.sum();
            prior_weights(min_sample_idr) = config::learning_rate;
            new_sample_id = min_sample_idr;   
        }
        else
        {
            VectorXf dist_vector = (gram_matrix.diagonal().array() + new_train_sample_norm -2*gram_vector.array()).max(0);
            float new_sample_min_dist;
            int closest_sample_to_new_sampler,closest_sample_to_new_samplec;
            new_sample_min_dist = dist_vector.minCoeff(&closest_sample_to_new_sampler,&closest_sample_to_new_samplec);

            float existing_samples_min_dist;
            int closest_existing_sample1,closest_existing_sample2;
            existing_samples_min_dist = distance_matrix.minCoeff(&closest_existing_sample1,&closest_existing_sample2);
        
            if (closest_existing_sample1 == closest_existing_sample2)
            {
                printf("Score matrix diagonal filled wrongly\n");
                exit(-1);
            }
            if(new_sample_min_dist < existing_samples_min_dist)
            {
                prior_weights = prior_weights*(1 - config::learning_rate);
                            
                // Set the position of the merged sample
                merged_sample_id = closest_sample_to_new_sampler;
                                
                Array4D<complex<float> > cn_existing_sample_to_merge(cn_samplesf.dim1,cn_samplesf.dim2,cn_samplesf.dim3,1);
                Array4D<complex<float> > hog_existing_sample_to_merge(hog_samplesf.dim1,hog_samplesf.dim2,hog_samplesf.dim3,1);

                cn_existing_sample_to_merge.getVector() = 
                            Map<MatrixXcf>(cn_samplesf.data,cn_samplesf.stride12*cn_samplesf.dim3,cn_samplesf.dim4).middleCols(closest_sample_to_new_sampler,1);
                hog_existing_sample_to_merge.getVector() = 
                            Map<MatrixXcf>(hog_samplesf.data,hog_samplesf.stride12*hog_samplesf.dim3,hog_samplesf.dim4).middleCols(closest_sample_to_new_sampler,1);
                
                // Merge the new_train_sample with existing sample
                merge_samples(cn_existing_sample_to_merge, hog_existing_sample_to_merge,cn_train_sample,hog_train_sample,
                                     prior_weights(closest_sample_to_new_sampler),config::learning_rate);
                
                // Update distance matrix and the gram matrix
                update_distance_matrix(gram_vector, new_train_sample_norm, merged_sample_id, -1, prior_weights(merged_sample_id), config::learning_rate);
                
                // Update the prior weight of the merged sample
                prior_weights(closest_sample_to_new_sampler) = prior_weights(closest_sample_to_new_sampler) + config::learning_rate;   
            }
            else
            {
                prior_weights = prior_weights*(1 - config::learning_rate);
                if(prior_weights(closest_existing_sample2) > prior_weights(closest_existing_sample1))
                {
                    int temp = closest_existing_sample1;
                    closest_existing_sample1 = closest_existing_sample2;
                    closest_existing_sample2 = temp;
                }
                Array4D<complex<float> > cn_sample_to_merge1(cn_samplesf.dim1,cn_samplesf.dim2,cn_samplesf.dim3,1);
                Array4D<complex<float> > hog_sample_to_merge1(hog_samplesf.dim1,hog_samplesf.dim2,hog_samplesf.dim3,1);
                Array4D<complex<float> > cn_sample_to_merge2(cn_samplesf.dim1,cn_samplesf.dim2,cn_samplesf.dim3,1);
                Array4D<complex<float> > hog_sample_to_merge2(hog_samplesf.dim1,hog_samplesf.dim2,hog_samplesf.dim3,1);

                cn_sample_to_merge1.getVector()
                        = Map<MatrixXcf>(cn_samplesf.data,cn_samplesf.stride12*cn_samplesf.dim3,cn_samplesf.dim4).middleCols(closest_existing_sample1,1);
                hog_sample_to_merge1.getVector()
                        = Map<MatrixXcf>(hog_samplesf.data,hog_samplesf.stride12*hog_samplesf.dim3,hog_samplesf.dim4).middleCols(closest_existing_sample1,1);
                cn_sample_to_merge2.getVector()
                        = Map<MatrixXcf>(cn_samplesf.data,cn_samplesf.stride12*cn_samplesf.dim3,cn_samplesf.dim4).middleCols(closest_existing_sample2,1);
                hog_sample_to_merge2.getVector()
                        = Map<MatrixXcf>(hog_samplesf.data,hog_samplesf.stride12*hog_samplesf.dim3,hog_samplesf.dim4).middleCols(closest_existing_sample2,1);

                merge_samples(cn_sample_to_merge1, hog_sample_to_merge1,cn_sample_to_merge2,hog_sample_to_merge2,
                                     prior_weights(closest_existing_sample1), prior_weights(closest_existing_sample2));
                        
                update_distance_matrix(gram_vector, new_train_sample_norm, closest_existing_sample1, closest_existing_sample2,
                                             prior_weights(closest_existing_sample1),  prior_weights(closest_existing_sample2)); 
                                             
                prior_weights(closest_existing_sample1) += prior_weights(closest_existing_sample2);
                prior_weights(closest_existing_sample2) = config::learning_rate;

                merged_sample_id = closest_existing_sample1;
                new_sample_id = closest_existing_sample2;  
            }
        }
    }
    else
    {
        update_distance_matrix(gram_vector,new_train_sample_norm,num_training_samples,-1,0,1);

        if(num_training_samples ==0)
            prior_weights(num_training_samples) = 1;
        else
        {
            prior_weights = prior_weights*(1 - config::learning_rate);
            prior_weights(num_training_samples) = config::learning_rate;
        }
        new_sample_id = num_training_samples; 
    }
    // Ensure that prior weights always sum to 1
    if (abs(1 - prior_weights.sum()) > 1e-5)
    {
        printf("Weights not properly updated\n");
        exit(0);
    }
    if(num_training_samples<config::nSamples)
        ++num_training_samples;
}

VectorXf SampleSpace::find_gram_vector(const Array4D<complex<float> >& cn_samplesf,const Array4D<complex<float> >& hog_samplesf,
                                        const Array4D<complex<float> >& cn_new_sample,const Array4D<complex<float> >& hog_new_sample)
{
    VectorXf gram_vector(config::nSamples);
    float* gram_vector_data = gram_vector.data();
    for(int i=0;i<gram_vector.size();++i)
        gram_vector_data[i] = INFINITY;
    VectorXf ip;
    if(num_training_samples >= config::nSamples)
    {
        ip = 2*(Map<MatrixXcf>(cn_samplesf.data,cn_samplesf.stride12*cn_samplesf.dim3,cn_samplesf.dim4).transpose() * 
                                Map<MatrixXcf>(cn_new_sample.data,cn_new_sample.size,1).conjugate()).real();
        ip +=  2*(Map<MatrixXcf>(hog_samplesf.data,hog_samplesf.stride12*hog_samplesf.dim3,hog_samplesf.dim4).transpose() * 
                                Map<MatrixXcf>(hog_new_sample.data,hog_new_sample.size,1).conjugate()).real();        
        gram_vector = ip;
    }
    else if(num_training_samples >0)
    {
        ip = 2*(Map<MatrixXcf>(cn_samplesf.data,cn_samplesf.stride12*cn_samplesf.dim3,cn_samplesf.dim4).leftCols(num_training_samples).transpose() * 
                                Map<MatrixXcf>(cn_new_sample.data,cn_new_sample.size,1).conjugate()).real();
        ip += 2*(Map<MatrixXcf>(hog_samplesf.data,hog_samplesf.stride12*hog_samplesf.dim3,hog_samplesf.dim4).leftCols(num_training_samples).transpose() * 
                                Map<MatrixXcf>(hog_new_sample.data,hog_new_sample.size,1).conjugate()).real();
        gram_vector.topRows(num_training_samples) = ip;
    }
    return gram_vector;
}

void SampleSpace::update_distance_matrix(VectorXf& gram_vector,float sample_norm, int id1, int id2, float w1, float w2)
{
    float alpha1 = w1/(w1+w2);
    float alpha2 = 1 - alpha1;

    if(id2<0)
    {
        float norm_id1 = gram_matrix(id1, id1);
        if(alpha1 == 0)
        {
            gram_matrix.middleCols(id1,1) = gram_vector;
            gram_matrix.middleRows(id1,1) = gram_vector.transpose();
            gram_matrix(id1, id1) = sample_norm;
        }
        else if(alpha2 == 0)
        {

        }
        else
        {
            gram_matrix.middleCols(id1,1) = alpha1*gram_matrix.middleCols(id1,1) + alpha2*gram_vector;
            gram_matrix.middleRows(id1,1) = gram_matrix.middleCols(id1,1).eval().transpose();
            gram_matrix(id1, id1) = alpha1*alpha1*norm_id1 + alpha2*alpha2*sample_norm + 2*alpha1*alpha2*gram_vector(id1);
        }
        // Update distance matrix
        distance_matrix.middleCols(id1,1) = vec_zero.array().max(gram_matrix.diagonal().array() + gram_matrix(id1, id1)- 2*gram_matrix.middleCols(id1,1).array());
        distance_matrix.middleRows(id1,1) = distance_matrix.middleCols(id1,1).eval().transpose() ;
        distance_matrix(id1,id1) = INFINITY;

    }
    else
    {    
        if(alpha1 == 0 || alpha2 == 0)
        {
            printf("Error!\n");
            exit(-1);
        }
         
        float norm_id1 = gram_matrix(id1, id1);
        float norm_id2 = gram_matrix(id2, id2);
        float ip_id1_id2 = gram_matrix(id1,id2);
        
        // Handle the merge of existing samples
        gram_matrix.middleCols(id1,1) = alpha1*gram_matrix.middleCols(id1,1) + alpha2*gram_matrix.middleCols(id2,1);
        gram_matrix.middleRows(id1,1) = gram_matrix.middleCols(id1,1).eval().transpose();
        gram_matrix(id1, id1) = alpha1*alpha1*norm_id1 + alpha2*alpha2*norm_id2 + 2*alpha1*alpha2*ip_id1_id2;        
        gram_vector(id1) = alpha1*gram_vector(id1) + alpha2*gram_vector(id2);

        // Handle the new sample
        gram_matrix.middleCols(id2,1) = gram_vector;
        gram_matrix.middleRows(id2,1) = gram_matrix.middleCols(id2,1).eval().transpose();
        gram_matrix(id2, id2) = sample_norm;

        
        // Update the distance matrix
        distance_matrix.middleCols(id1,1) = vec_zero.array().max(gram_matrix.diagonal().array() + gram_matrix(id1, id1) - 2*gram_matrix.middleCols(id1,1).array());
        distance_matrix.middleRows(id1,1) = distance_matrix.middleCols(id1,1).eval().transpose();
        distance_matrix(id1,id1) = INFINITY;
        
        distance_matrix.middleCols(id2,1) = vec_zero.array().max(gram_matrix.diagonal().array() + gram_matrix(id2, id2) - 2*gram_matrix.middleCols(id2,1).array());
        distance_matrix.middleRows(id2,1) = distance_matrix.middleCols(id2,1).eval().transpose();
        distance_matrix(id2,id2) = INFINITY;
    }
}

void SampleSpace::merge_samples(const Array4D<complex<float> >& cn_sample1,const Array4D<complex<float> >& hog_sample1,
                                    const Array4D<complex<float> >& cn_sample2,const Array4D<complex<float> >& hog_sample2,
                                     float w1, float w2)
{   // 这里可以将 生成 与 更新 合并
    // Normalise the weights so that they sum to one
    float alpha1 = w1/(w1+w2);
    float alpha2 = 1 - alpha1;

    switch(config::sample_merge_type)
    {
        case replace:
            cn_merged_sample = cn_sample1;
            hog_merged_sample = hog_sample1;
        break;
        case merge:
            cn_merged_sample.resize(cn_sample1.dim1,cn_sample1.dim2,cn_sample1.dim3,cn_sample1.dim4);
            Map<MatrixXcf>(cn_merged_sample.data,1,cn_sample1.size).array()
                    = Map<MatrixXcf>(cn_sample1.data,1,cn_sample1.size).array() * alpha1 + 
                        Map<MatrixXcf>(cn_sample2.data,1,cn_sample2.size).array() *alpha2;

            hog_merged_sample.resize(hog_sample1.dim1,hog_sample1.dim2,hog_sample1.dim3,hog_sample1.dim4);
            Map<MatrixXcf>(hog_merged_sample.data,1,hog_sample1.size).array()
                    = Map<MatrixXcf>(hog_sample1.data,1,hog_sample1.size).array() * alpha1 + 
                        Map<MatrixXcf>(hog_sample2.data,1,hog_sample2.size).array() *alpha2;
        break;
        default:
            printf("Invalid sample merge type\n");
            exit(0);
            break;
    }
}

}