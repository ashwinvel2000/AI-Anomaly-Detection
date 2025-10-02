using Microsoft.AspNetCore.Mvc;

namespace taqa.polaris.sit.Controllers
{
    [Route("AnomalyDetector")]
    public class AnomalyDetectorController : Controller
    {
        [HttpGet("")]
        public IActionResult Index() => View();
    }
}
