$.fn.extend({
  animateCss: function(animationName, callback) {
    var animationEnd = (function(el) {
      var animations = {
        animation: 'animationend',
        OAnimation: 'oAnimationEnd',
        MozAnimation: 'mozAnimationEnd',
        WebkitAnimation: 'webkitAnimationEnd',
      };

      for (var t in animations) {
        if (el.style[t] !== undefined) {
          return animations[t];
        }
      }
    })(document.createElement('div'));

    this.addClass('animated ' + animationName).one(animationEnd, function() {
      $(this).removeClass('animated ' + animationName);

      if (typeof callback === 'function') callback();
    });

    return this;
  },
});

var followers_count, friends_count, listed_count, statuses_count;
var progressBar = $("#progress");

function resetAvatars() {
	$(".Avatar.imground").attr('src', 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png');
	$(".Profile.imground").attr('src', 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png');
}
function loadUserInfo() {
    const response = JSON.parse(this.responseText);
	if(response["id"]) {
		$(".Avatar.imground").attr('src', response["profile_image_url_https"]);
		$(".Profile.imground").attr('src', response["profile_image_url_https"]);
		$('.post-wrapper').animateCss('fadeInDown', function() {
			$(".TweetBoxToolbar button").prop('disabled', false); //disable
			$(".tweet-box form .post").prop('disabled', false); //disable
		});	
		followers_count = response["followers_count"];
		friends_count = response["friends_count"];
		listed_count = response["listed_count"];
		statuses_count = response["statuses_count"];
		
	} else {
		$('.post-wrapper').animateCss('fadeOutUp');
		$(".Profile.imground").addClass('itemblur');
		setTimeout(function(){
			$(".Profile.imground").removeClass('itemblur');
		},500);
	}
	$( ".loader" ).toggle(false);
}
function requestUserInfo() {
	var request = new XMLHttpRequest();
	$( ".loader" ).toggle(true);
	$(".TweetBoxToolbar button").prop('disabled', true); //disable
	$(".tweet-box form .post").prop('disabled', true); //disable
	resetAvatars();
    request.addEventListener('load', loadUserInfo);
    request.open('GET', '/account?post=' + encodeURIComponent($("#search-input")[0].value));
    request.send();
}
function loadBestDate() {
	const data = JSON.parse(this.responseText);
	var results = $("#results");
	var ctx = $("#Charts");
	results.children("h3").remove();
	results.prepend( "<h3>Max Score : "+data["max"][0]["max_score"]+", "+data["max"][1]["max_score"]+" </h3>" );
	console.log(data);
	var myChart = new Chart(ctx, {
		type: 'line',
		data: {
			labels: data["labels"],
			datasets: [{
				backgroundColor: 'rgba(225, 0, 0, 0.1)',
				label: 'Sunday',
				data: data[0][0]
			},{
				backgroundColor: 'rgba(0, 0, 255, 0.1)',
				label: 'Monday',
				data: data[1][0]
			},{
				backgroundColor: 'rgba(0, 255, 0, 0.1)',
				label: 'Tuesday',
				data: data[2][0]
			},{
				backgroundColor: 'rgba(0, 255, 255, 0.1)',
				label: 'Wednesday',
				data: data[3][0]
			},{
				backgroundColor: 'rgba(255, 255, 0, 0.1)',
				label: 'Thursday',
				data: data[4][0]
			},{
				backgroundColor: 'rgba(15, 255, 15, 0.1)',
				label: 'Friday',
				data: data[5][0]
			},{
				backgroundColor: 'rgba(15, 15, 150, 0.1)',
				label: 'Saturday',
				data: data[6][0]
			}]
		},
		options: {
			responsive: true,
			scales: {
				xAxes: [{
					type: 'linear',
					position: 'bottom'
				}]
			}
		}
	});
	$("#results").fadeIn();
	$(".wrapper").hide();
	$(".TweetBoxToolbar button").prop('disabled', false); 
	$(".tweet-box form .post").prop('disabled', false); 
}
function requestBestDate() {
    var request = new XMLHttpRequest();
	$("#results").hide();
	$(".wrapper").show();
    request.addEventListener('load', loadBestDate);
	request.open('GET', '/ask?followers_count=' + encodeURIComponent(followers_count) + '&friends_count=' + encodeURIComponent(friends_count) + '&listed_count=' + encodeURIComponent(listed_count) + '&statuses_count=' + encodeURIComponent(statuses_count) + '&text=' + encodeURIComponent($("#post-input")[0].value), true);
    request.send();
	$(".TweetBoxToolbar button").prop('disabled', true); //disable
	$(".tweet-box form .post").prop('disabled', true); //disable
}
function loadScore() {
	const data = JSON.parse(this.responseText);
	var results = $(".TweetBoxToolbar");
	results.children("h3").remove();
	results.prepend( "<h3>Score of the current text : "+ Math.round(data[0]) +"</h3>" );	
	$(".TweetBoxToolbar button").prop('disabled', false); 
	$(".tweet-box form .post").prop('disabled', false); 
}
function requestScore() {
    var request = new XMLHttpRequest();
    request.addEventListener('load', loadScore);
	request.open('GET', '/score?followers_count=' + encodeURIComponent(followers_count) + '&friends_count=' + encodeURIComponent(friends_count) + '&listed_count=' + encodeURIComponent(listed_count) + '&statuses_count=' + encodeURIComponent(statuses_count) + '&text=' + encodeURIComponent($("#post-input")[0].value), true);
    request.send();
	$(".TweetBoxToolbar button").prop('disabled', true); //disable
	$(".tweet-box form .post").prop('disabled', true); //disable
}

$("#proposedate").click(requestBestDate);
$("#predictscore").click(requestScore);
$("#magnifier").click(requestUserInfo);
$("#search-input").keypress(function(event) {
    if (event.key === "Enter") {
        requestUserInfo();
    }
});
 $(".TweetBoxToolbar button").prop('disabled', true); //disable
 $(".tweet-box form .post").prop('disabled', true); //disable
resetAvatars();
$('.login').animateCss('bounce');