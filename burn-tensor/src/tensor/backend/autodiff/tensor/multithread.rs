#[cfg(test)]
mod tests {
    use crate::{backend::autodiff::helper::ADTchTensor, Data, TensorBase, TensorOpsMatmul};

    #[test]
    fn should_behave_the_same_with_multithread() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f32, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);

        let with_move = || {
            let tensor_1 = ADTchTensor::from_data(data_1.clone());
            let tensor_2 = ADTchTensor::from_data(data_2.clone());

            let tensor_3 = tensor_1.matmul(&tensor_2);
            let tensor_4 = tensor_3.matmul(&tensor_2);
            let tensor_5 = tensor_4.matmul(&tensor_3);

            let tensor_1_cloned = tensor_1.clone();
            let tensor_2_cloned = tensor_2.clone();
            let tensor_5_cloned = tensor_5.clone();

            let first_call = move || {
                let tensor_6_1 = tensor_5_cloned.matmul(&tensor_2_cloned);
                tensor_6_1.matmul(&tensor_1_cloned)
            };

            let tensor_1_cloned = tensor_1.clone();
            let tensor_2_cloned = tensor_2.clone();
            let tensor_5_cloned = tensor_5.clone();

            let second_call = move || {
                let tensor_6_2 = tensor_5_cloned.matmul(&tensor_1_cloned);
                tensor_6_2.matmul(&tensor_2_cloned)
            };

            let tensor_7_1_handle = std::thread::spawn(first_call);
            let tensor_7_2_handle = std::thread::spawn(second_call);

            let tensor_7_1 = tensor_7_1_handle.join().unwrap();
            let tensor_7_2 = tensor_7_2_handle.join().unwrap();
            let tensor_8 = tensor_7_1.matmul(&tensor_7_2);

            let grads = tensor_8.backward();

            let grad_1 = grads.wrt(&tensor_1).unwrap();
            let grad_2 = grads.wrt(&tensor_2).unwrap();

            return (grad_1.clone(), grad_2.clone());
        };
        let without_move = || {
            let tensor_1 = ADTchTensor::from_data(data_1.clone());
            let tensor_2 = ADTchTensor::from_data(data_2.clone());

            let tensor_3 = tensor_1.matmul(&tensor_2);
            let tensor_4 = tensor_3.matmul(&tensor_2);
            let tensor_5 = tensor_4.matmul(&tensor_3);

            // Task 1
            let tensor_6_1 = tensor_5.matmul(&tensor_2);
            let tensor_7_1 = tensor_6_1.matmul(&tensor_1);

            // Task 2
            let tensor_6_2 = tensor_5.matmul(&tensor_1);
            let tensor_7_2 = tensor_6_2.matmul(&tensor_2);

            let tensor_8 = tensor_7_1.matmul(&tensor_7_2);

            let grads = tensor_8.backward();

            let grad_1 = grads.wrt(&tensor_1).unwrap();
            let grad_2 = grads.wrt(&tensor_2).unwrap();

            return (grad_1.clone(), grad_2.clone());
        };

        let (grad_1, grad_2) = without_move();
        let (grad_1_moved, grad_2_moved) = with_move();

        assert_eq!(grad_1.to_data(), grad_1_moved.to_data());
        assert_eq!(grad_2.to_data(), grad_2_moved.to_data());
        assert_eq!(
            grad_1.to_data(),
            Data::from([
                [12255630000000.0, 8076727000000.0],
                [10450690000000.0, 8704954000000.0]
            ])
        );
        assert_eq!(
            grad_2.to_data(),
            Data::from([
                [17459151000000.0, 15028745000000.0],
                [14354680000000.0, 11433980000000.0]
            ])
        );
    }
}