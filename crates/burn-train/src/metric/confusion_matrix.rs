use burn_core::tensor::{Tensor, Int, ElementConversion};
use burn_core::tensor::backend::Backend;


#[derive(Debug, Clone)]
pub struct ConfusionMatrix <B: Backend> {
    pub matrix: Tensor<B, 2, Int>,
}

impl <B: Backend> ConfusionMatrix <B> {
    pub fn new() -> Self {
        let device = Default::default();
        let matrix: Tensor<B, 2, Int> = Tensor::zeros([1,1], &device);
        Self {matrix}
    }

    pub fn from_outputs(&mut self, outputs: &Tensor<B, 2>, targets: &Tensor<B, 1, Int>) {
        let device = Default::default();
        let [batch_size, _n_classes] = outputs.dims();     
        let predictions = outputs
            .clone()
            .argmax(1)
            .to_device(&device)
            .reshape([batch_size]); 


        let ee = predictions
                .clone()
                .equal(targets.clone())
                .argwhere()
                .to_data();

        //let classes = Tensor::<Backend,1,Int>::from_data([0], &device);
        //let tensor_updated = tensor.slice_assign(2..3, 2..3, tensor.slice(2..3, 2..3) + value);
        let mut matrix = Tensor::<B, 2, Int>::zeros([_n_classes,_n_classes], &device);

        let mut upd_mat = matrix.clone();

        for j in 0.. batch_size {
            let cls_pred = i64::from_elem(predictions.clone().to_data().value[j]) as usize;
            let cls_tgt = i64::from_elem(targets.clone().to_data().value[j]) as usize;

            if cls_pred == cls_tgt {
                let values = matrix
                    .clone()
                    .slice([cls_pred..cls_pred+1, cls_tgt..cls_tgt+1]) + 1;

                upd_mat = matrix
                    .slice_assign( [cls_pred..cls_pred+1, cls_tgt..cls_tgt+1], values); 
            }
            else {
                let values = matrix
                    .clone()
                    .slice([cls_pred..cls_pred+1, cls_tgt..cls_tgt+1]) + 1;

                upd_mat = matrix
                    .slice_assign( [cls_pred..cls_pred+1, cls_tgt..cls_tgt+1], values);                 
            }

            matrix = upd_mat.clone();
        
        }
        self.matrix = matrix;
        //println!("{}", self.matrix);
    }

    pub fn set(&mut self, t: Tensor::<B,2,Int>) {
        self.matrix = t;
    }

        /// Split confusion matrix in N one-vs-all binary confusion matrices
    pub fn split_cm(&self) -> Vec<ConfusionMatrix<B>> {
        let device = Default::default();
        let [dim1, _dim2] = self.matrix.dims();
        let sum = self.matrix.clone().sum();

        (0..dim1)
            .map(|i| {
                let tp = i32::from_elem(self.matrix
                    .clone()
                    .slice([i..i+1, i..i+1])
                    .to_data()
                    .value[0]);

                let fp = i32::from_elem(self.matrix
                    .clone()
                    .slice([i..i+1, 0..dim1]).sum().to_data().value[0])  - tp;

               let _fn = i32::from_elem(self.matrix
                .clone()
                .slice([0..dim1, i..i+1]).sum().to_data().value[0]) - tp;

                let tn = i32::from_elem(sum.to_data().value[0]) - tp - fp - _fn;

                let matrix = 
                    Tensor::<B, 2, Int>::from_data([[tp, fp],[_fn, tn]], &device);

                println!("{}", matrix);
                
                let mut cm = ConfusionMatrix::<B>::new();
                cm.set(matrix.clone());

                cm
            })
            .collect()
        }
}