 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
    
#include <cusp/graph/detail/host/maximum_flow.h>
#include <cusp/graph/detail/device/maximum_flow.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type 
maximum_flow(const MatrixType& G, ArrayType& flow, const IndexType src, const IndexType sink, cusp::host_memory)
{
    return cusp::graph::detail::host::maximum_flow(G, flow, src, sink);
}

//////////////////
// Device Paths //
//////////////////
template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type 
maximum_flow(const MatrixType& G, ArrayType& flow, const IndexType src, const IndexType sink, cusp::device_memory)
{
    return cusp::graph::detail::device::maximum_flow(G, flow, src, sink);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace graph
} // end namespace cusp

