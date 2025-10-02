using Microsoft.AspNetCore.Mvc;
using taqa.polaris.sit.Models;

namespace taqa.polaris.sit.Controllers
{
    public class ReportLandingController : Controller
    {
        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Index(ReportInputModel model)
        {
            if (ModelState.IsValid)
            {
                TempData["Message"] = "Report generation triggered successfully!";
                return RedirectToAction("Success");
            }
        
            return View(model); 
        }
    }
}
