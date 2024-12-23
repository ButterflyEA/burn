use crate::{element::JitElement, kernel, tensor::JitTensor, JitRuntime};
use burn_tensor::{Shape, TensorData};
use cubecl::CubeElement;

pub(crate) fn from_data<R: JitRuntime, E: JitElement>(
    data: TensorData,
    device: &R::Device,
) -> JitTensor<R> {
    let shape: Shape = (&data.shape).into();
    let client = R::client(device);
    let buffer = client.create(data.convert::<E>().as_bytes());

    JitTensor::new_contiguous(client, device.clone(), shape, buffer, E::dtype())
}

pub(crate) async fn into_data<R: JitRuntime, E: JitElement>(tensor: JitTensor<R>) -> TensorData {
    let tensor = kernel::into_contiguous(tensor);

    let bytes = tensor.client.read_one_async(tensor.handle.binding()).await;
    TensorData::new(E::from_bytes(&bytes).to_vec(), tensor.shape)
}

#[allow(unused, reason = "useful for debugging kernels")]
pub(crate) fn into_data_sync<R: JitRuntime, E: JitElement>(tensor: JitTensor<R>) -> TensorData {
    let tensor = kernel::into_contiguous(tensor);

    let bytes = tensor.client.read_one(tensor.handle.binding());
    TensorData::new(E::from_bytes(&bytes).to_vec(), tensor.shape)
}

pub(crate) async fn bool_into_data<R: JitRuntime>(tensor: JitTensor<R>) -> TensorData {
    let tensor = kernel::into_contiguous(tensor);
    let bytes = tensor.client.read_one_async(tensor.handle.binding()).await;
    TensorData::new(
        u32::from_bytes(&bytes).iter().map(|i| *i != 0).collect(),
        tensor.shape,
    )
}

pub(crate) fn to_device<R: JitRuntime>(tensor: JitTensor<R>, device: &R::Device) -> JitTensor<R> {
    if &tensor.device == device {
        return tensor;
    }

    let client = R::client(device);
    tensor.to_client(client, device.clone())
}

pub(crate) fn empty<R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
) -> JitTensor<R> {
    let client = R::client(device);
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    JitTensor::new_contiguous(client, device.clone(), shape, buffer, E::dtype())
}

pub(crate) fn swap_dims<R: JitRuntime>(
    mut tensor: JitTensor<R>,
    dim1: usize,
    dim2: usize,
) -> JitTensor<R> {
    tensor.strides.swap(dim1, dim2);
    tensor.shape.dims.swap(dim1, dim2);

    tensor
}

pub(crate) fn permute<R: JitRuntime>(mut tensor: JitTensor<R>, axes: &[usize]) -> JitTensor<R> {
    // remap strides
    tensor.strides = axes.iter().map(|i| tensor.strides[*i]).collect();

    // remap shape
    tensor.shape.dims = axes.iter().map(|i| tensor.shape.dims[*i]).collect();

    tensor
}
pub(crate) fn expand<R: JitRuntime>(tensor: JitTensor<R>, target_shape: Shape) -> JitTensor<R> {
    let ndims_in = tensor.shape.num_dims();
    let ndims_out = target_shape.num_dims();

    // Initialize new strides with zeros
    let mut new_strides = vec![0usize; ndims_out];

    // Calculate the difference in dimensions
    let dim_diff = ndims_out.saturating_sub(ndims_in);

    // Compare dimensions from the end, setting strides for matching dimensions or broadcasted ones
    let mut tensor_dim_iter = tensor.shape.dims.iter().rev();
    for i in (0..ndims_out).rev() {
        if i >= dim_diff {
            if let Some(&tensor_dim) = tensor_dim_iter.next() {
                if tensor_dim == target_shape.dims[i] || tensor_dim == 1 {
                    // Copy stride for non-broadcast dimensions or set to 0 for broadcast ones
                    new_strides[i] = if tensor_dim == target_shape.dims[i] {
                        tensor.strides[i - dim_diff]
                    } else {
                        0
                    };
                } else {
                    // Error handling: Dimension mismatch for broadcasting
                    panic!(
                        "Dimension mismatch: cannot broadcast dimension {} of tensor to target shape",
                        tensor_dim
                    );
                }
            } else {
                // If the input tensor has fewer dimensions, treat missing dimensions as 1
                // and set stride to 0 (broadcasting)
                new_strides[i] = 0;
            }
        } else {
            // For extra dimensions in the target shape, set stride to 0 (broadcasting)
            new_strides[i] = 0;
        }
    }

    JitTensor {
        client: tensor.client,
        device: tensor.device,
        shape: target_shape,
        strides: new_strides,
        handle: tensor.handle,
        dtype: tensor.dtype,
    }
}

pub(crate) fn reshape<R: JitRuntime>(tensor: JitTensor<R>, shape: Shape) -> JitTensor<R> {
    // TODO: Not force standard layout all the time (improve performance).
    let tensor = kernel::into_contiguous(tensor);

    JitTensor::new_contiguous(
        tensor.client,
        tensor.device,
        shape,
        tensor.handle,
        tensor.dtype,
    )
}
