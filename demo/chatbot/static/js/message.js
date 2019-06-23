(function () {
    var Message;
	var messages = [];
    Message = function (arg) {
        this.text = arg.text, this.message_side = arg.message_side;
		this.dialog_act = arg.dialog_act
		this.id = messages.length;
		this.showDialogAct = function(_this) {
			return function() {
				console.log(_this.dialog_act);
				$('#message_' + _this.id).find('.text').html(_this.dialog_act);
			};
		}(this);
		this.showText = function(_this) {
			return function() {
				$('#message_' + _this.id).find('.text').html(_this.text);
			};
		}(this);
        this.draw = function (_this) {
            return function () {
                var $message;
                $message = $($('.message_template').clone().html());
				$message.attr('id', 'message_' + _this.id)
				if ($('#mode').attr('mode') == 'text') {
					$message.addClass(_this.message_side).find('.text').html(_this.text);
				} else {
					$message.addClass(_this.message_side).find('.text').html(_this.dialog_act);
				}
                $('.messages').append($message);
                return setTimeout(function () {
                    return $message.addClass('appeared');
                }, 0);
            };
        }(this);
        return this;
    };
    $(function () {
        var getMessageText, message_side, sendMessage;
        getMessageText = function () {
            var $message_input;
            $message_input = $('.message_input');
            return $message_input.val();
        };
		showText = function() {
			messages.forEach(function (message) {
				message.showText()
			});
		}
		showDialogAct = function() {
			messages.forEach(function (message) {
				message.showDialogAct()
			});
		}
		showMessage = function(text, dialog_act, message_side) {
            var $messages = $('.messages');
            message = new Message({
                text: text,
                message_side: message_side,
				dialog_act: dialog_act
            });
            message.draw();
			messages.push(message);
            return $messages.animate({ scrollTop: $messages.prop('scrollHeight') }, 300);
		}
        sendMessage = function (text) {
            var message;
            if (text.trim() === '') {
                return;
            }
            $('.message_input').val('');
			$.get('/chatbot/send_message', {'user_input': text}, function(ret) {
				ret = JSON.parse(ret);
				showMessage(ret['user_text'], ret['user_action'], 'right');
				setTimeout(function (){
					showMessage(ret['agent_text'], ret['agent_action'], 'left');
				}, 500);
			})
        };
		reset = function () {
			$('.messages').empty();
			messages = []
			$.get('/chatbot/reset')
			showMessage('Hello, may I help you?', '[{intent: greeting}]', 'left');
		}
        $('.send_message').click(function (e) {
            return sendMessage(getMessageText());
        });
        $('.message_input').keyup(function (e) {
            if (e.which === 13) {
                return sendMessage(getMessageText());
            }
        });
		$('#show_table').click(function(e) {
			if ($('#timetable').is(':hidden')) {
				$('#timetable').show();
				$('.chat_window').css('width', '50%').css('width', '-=20px');
				$('.chat_window').css('transform', 'translateX(-100%) translateY(-50%)');
			} else {
				$('#timetable').hide();
				$('.chat_window').css('width', '100%').css('width', '-=20px');
				$('.chat_window').css('transform', 'translateX(-50%) translateY(-50%)');
			}
		})
		$('#reset').click(function(e) {
			reset();
		})
		$('#mode').click(function(e) {
			if ($('#mode').attr('mode') == 'dia_act') {
				$('#mode').attr('mode', 'text');
				$('#mode').find('.text').text('Show Dialog Act');
				showText();
			} else {
				$('#mode').attr('mode', 'dia_act');
				$('#mode').find('.text').text('Show Text');
				showDialogAct();
			}
		})
		reset();
		$('#timetable').hide();
		$('.chat_window').css('width', '100%').css('width', '-=20px');
		$('.chat_window').css('transform', 'translateX(-50%) translateY(-50%)');
		/*
        sendMessage('Hello Philip! :)');
        setTimeout(function () {
            return sendMessage('Hi Sandy! How are you?');
        }, 1000);
        return setTimeout(function () {
            return sendMessage('I\'m fine, thank you!');
        }, 2000);
		*/
    });
}.call(this));
